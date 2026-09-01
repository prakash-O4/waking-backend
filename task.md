# AGENT-23 — Redaction-approval guard, source_publication backfill, canonical content_hash

## Branch
`agent/small-safety-fixes` (base: `dev`, cut alongside AGENT-22 — both touch
`app/ingestion/pipeline.py` but at unrelated, non-adjacent functions
(`_content_hash` near the top vs. AGENT-22's CHUNK-stage/`_commence_date`
changes); if a merge conflict shows up anyway, that's on Claude to resolve
at merge time, not yours to avoid by waiting)

## Objective
Three independent, small, confirmed gaps from the 2026-09-01 follow-up
review. Unrelated to each other and to AGENT-22 — bundled into one task
because each is too small to justify its own branch/review cycle on its
own, not because they share a theme.

## Scope

### A. `scripts/review_documents.py` — block approval on failed redaction
`_approve_one()` checks `ingestion_status`/distinct-approver only —
`documents.redaction_failed` is displayed by `list_pending()` but never
blocks approval. Add it to the initial `SELECT ... FOR UPDATE` and refuse
approval outright (before touching approver state) if `TRUE`, with a clear
message (e.g. "redaction verification failed, cannot approve"). Not
currently live-exploitable (only NKP ingestion ever sets this column, and
NKP retrieval is hard-disabled since AGENT-19) but the tool itself
shouldn't allow it regardless of what's wired to retrieval today.

### B. Backfill `source_publication.kind` for pre-AGENT-20 rows
`upsert_source()`'s `SELECT ... LIMIT 1` early-return means the
`derived_verified` fix (AGENT-20) only applies to rows inserted after it
merged — every `source_publication` row written before that still says
`official_copy_unverified`. New small script (e.g.
`scripts/backfill_source_kind.py`) doing exactly one idempotent statement:
`UPDATE source_publication SET kind='derived_verified' WHERE
kind='official_copy_unverified'`. Confirmed safe blanket-scope today: grep
shows `upsert_source()` is the only writer of this table, called only from
law ingestion. `--dry-run` (report count) + real run, matching this repo's
established backfill-script convention.

### C. `pipeline.py::_content_hash()` — match the documented spec
`docs/ingestion_design.md`'s idempotency section specifies
`content_hash = sha256(NFC + digit-fold + whitespace-canonicalized raw
content)`. Current code only does NFC. Add digit-fold (reuse the pattern
in `pii_redactor.py::_digit_fold` — Devanagari→ASCII via
`str.translate`, comparison-only, don't change what's stored) and
whitespace canonicalization (collapse runs of whitespace — spaces, tabs,
newlines — to a single space; strip leading/trailing whitespace from the
result). This is a hashing-input transform only — never changes
`raw_content`/`chunk_text` as stored or rendered.
**Required companion, not optional**: changing the hash function changes
the computed hash for effectively every already-ingested document even
though the source text hasn't changed. Left alone, the next
`ingest_laws.py` run would read every one of them as "amended" and reset
`ingestion_status` to `pending` + clear both approvers — for zero real
content change. Write a small new script (e.g.
`scripts/recompute_content_hashes.py`) that, for each already-ingested
`documents` row matched to its `laws.jsonl` record by `source_id`,
recomputes the hash with the fixed function and, only where it differs
from the stored value, `UPDATE documents SET content_hash=%s` **directly**
— must NOT touch `ingestion_status`, `approved_by`, or
`second_approved_by`. `--dry-run` + real run, per-document commit/rollback
(same discipline as `backfill_authority_layer.py`, but keep this as its
own script rather than extending that one — AGENT-22 is already extending
it for something unrelated, and this task should stay file-disjoint from
that one wherever it can).

## Allowed scope
- `scripts/review_documents.py`, `tests/test_review_documents.py`
- `scripts/backfill_source_kind.py` (new), its test
- `app/ingestion/pipeline.py` (only `_content_hash()`)
- `scripts/recompute_content_hashes.py` (new), its test

## Forbidden
- Do not touch `app/retrieval/validation_gate.py`, `chunks`/`documents`
  schema beyond what's already there, or anything in AGENT-22's allowed
  scope — separate task, avoid overlap.
- Do not change what `documents.raw_content`/`chunks.chunk_text` store —
  the canonicalization in item C is a hash-input transform only.
- Do not extend `backfill_authority_layer.py` for item C's recompute —
  keep it a separate script (see item C).

## Required checks
- `make test`
- `make lint` (`scripts/review_documents.py` isn't in the fixed lint file
  list — same pre-existing gap as `review_lifecycle.py`; run ruff + mypy
  --strict on it and the two new scripts manually and report the result)
- `make eval-gates`

## Self-review before returning
- Confirm item A's guard fires before any `UPDATE` runs, not after —
  refusing must never leave `approved_by` partially set.
- Confirm item B's `UPDATE` is idempotent (running it twice does nothing
  the second time).
- Confirm item C's recompute script never flips `ingestion_status` even
  when the hash changes — that's the entire point of a separate script
  instead of just re-running `ingest_laws.py`.
- Run item B and item C's `--dry-run` against whatever local corpus state
  you have and report before/after counts.

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any
AI tool. No `Co-Authored-By: Claude` trailer, no "Generated with Claude" line.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
