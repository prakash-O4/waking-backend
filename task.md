# AGENT-22 — Wire citation rendering + temporal revalidation to real authority

**Re-dispatch note**: this task was assigned once before and the report
that came back described this work in detail, but none of it was actually
in the repo — a separate task's implementation (AGENT-23) was sitting
uncommitted on this branch's working tree instead and got reported under
this task's name. If you're running this in a working directory shared
with another session, commit your work on *this* branch before switching
away from it, and don't report completion for a branch you haven't
actually run `git diff`/`git status` against immediately beforehand.

## Branch
`agent/authority-linked-citations` (base: `dev`)

## Objective
Confirmed by re-reading the code directly (2026-09-01): `validation_gate.py`'s
`_citation()`/`_expression()` resolve every claim purely against `chunks` — a
search derivative — and never touch `component`/`source_publication`/
`lifecycle_effect`. `docs/ingestion_design.md`'s own migration-005 header
says outright: *"chunks and its embedding index are a SEARCH DERIVATIVE...
Every citation is revalidated against the authority store before it ships."*
That revalidation doesn't happen. Two concrete consequences, both real:

1. **No live temporal check against `lifecycle_effect`.** `chunks.effective_date_ad`
   is a cache populated once at ingest time from an approved `commence`
   effect (PS-2, correct for that purpose). But nothing ever checks whether
   a *later* approved `repeal`/`expiry`/`suspend` effect has since taken the
   component out of force. A chunk ingested before a later repeal stays
   "eligible" forever via its stale cache. This is the actual gap behind
   Core Invariant #6 / §7.7's *"verify temporal validity at the claim's
   as-of → revalidate vs Postgres"* — eligibility gate (AGENT-19/20) only
   handles the commence side; nothing handles the termination side.
2. **No real provenance.** `_citation()`'s `source_kind` reads
   `chunks.source_type` (act/regulation/nkp_case — a different concept),
   and `ocr_confidence` is hardcoded `None`. Neither is close to
   `source_publication.kind`/`documents.ocr_confidence`, the actual PS-10
   fields.

Root cause for both: **chunks have no way to reach the authority tables.**
No `component_uri` on `chunks`, no `source_pub_id` on `documents` (even
though `pipeline.py::ingest_law()` already computes one locally and
discards it). Fixing this needs two small schema additions — both just
persisting values the pipeline already computes, not new extraction logic.

## Scope

### 1. Migration `migrations/011_chunk_authority_links.sql`
```sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS component_uri TEXT;
CREATE INDEX IF NOT EXISTS chunks_component_uri_idx
    ON chunks (component_uri) WHERE component_uri IS NOT NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_pub_id UUID
    REFERENCES source_publication(id);
```
Register as `("011_chunk_authority_links", "011_chunk_authority_links.sql")`
in `scripts/migrate.py`'s `MIGRATIONS` list, after `010_...`.

### 2. `pipeline.py::ingest_law()`
- Persist `source_pub_id` (the `PERSIST_AUTHORITY` stage already computes
  this into a local variable and only threads it into the lifecycle
  extractors) onto the `documents` row: `UPDATE documents SET
  source_pub_id=%s WHERE id=%s`.
- Populate `chunk.component_uri` for every law/regulation chunk before
  `EMBED_AND_UPSERT`. Extract the URI-construction logic already living
  inside `_commence_date()` (`f"{work_uri}/dafa/{section_number.translate(
  _DEVANAGARI_DIGITS)}"`) into a small shared helper — e.g.
  `_component_uri(work_uri: str, section_number: str) -> str` — used by
  both `_commence_date()` and the new population code, instead of a second
  copy of the same string-building. Use `chunk.section_number` when
  `chunk.level == 'section'`; use `chunk.parent_section` for
  `'subsection'`/`'proviso'` chunks (both resolve to the same enclosing
  दफा — matches how `_commence_date()` already treats every chunk of a
  दफा as sharing one commence status). `chunk.level in ('act', 'chapter')`
  or `nkp_case` documents: leave `component_uri` NULL — no single
  component applies.
- `pgvector_indexer.py::upsert_document`/chunk insert needs a
  `component_uri` column added to whatever it writes per chunk — check
  the current insert statement and thread it through.

### 3. `app/retrieval/validation_gate.py`
- `_citation()`: join `chunks.document_id → documents.source_pub_id →
  source_publication.kind` for the real `source_kind`, and
  `documents.ocr_confidence` directly (same join). If `source_pub_id IS
  NULL` (nkp_case, or a pre-migration row not yet backfilled), fall back
  to today's `chunks.source_type`-based value rather than returning
  nothing — don't regress existing behavior for rows this migration
  hasn't reached yet.
- New temporal-authority check (name it, e.g., `_terminated_before`):
  given `chunks.component_uri` and the claim's `as_of`, query
  `lifecycle_effect` for any row with `approval_status='approved'`,
  `effect_type IN ('repeal', 'expiry', 'suspend')`, whose
  `legal_valid_time`'s lower bound is `<= as_of` (i.e. already in effect
  by that date). If found, the claim abstains (`citation = None`)
  regardless of what the hash check or `eligible_chunk_ids` said — this
  is the actual "revalidate against authority" enforcement point. A
  terminating effect whose `legal_valid_time` starts *after* the claim's
  `as_of` must **not** cause abstention — a claim about 2070 law is not
  affected by a 2080 repeal (Core Invariant #6, per-claim as-of). Chunks
  with `component_uri IS NULL` skip this check (nothing to look up) —
  note this as a residual gap the backfill (below) is what actually
  closes for the existing corpus.
- Do **not** replace `chunk_text`/`span_sha256` as the rendered text or
  its integrity hash, and do not attempt to cross-check it against
  `expression.text_hash` — different granularity (a chunk can be a
  subsection/proviso split of a full-दफा `expression`), and the existing
  hash check is already doing its designed job: §9 explicitly
  distinguishes *"integrity ≠ temporal validity; never new validation"* —
  the self-consistency hash check is the integrity half, done correctly
  already; this task adds the missing temporal-validity half. Don't
  conflate the two.

### 4. Backfill — extend `scripts/backfill_authority_layer.py`
It already iterates `documents` matched to their `laws.jsonl` record by
`source_id`, re-parses via `parse_law()`, and calls
`upsert_component`/`upsert_expression`/`upsert_source`. Extend it, per
document already being processed:
- `UPDATE documents SET source_pub_id=%s` using the `source_pub_id` its
  existing `upsert_source()` call already returns.
- For each parsed component: `UPDATE chunks SET component_uri=%s WHERE
  document_id=%s AND (section_number=%s OR parent_section=%s)` (matching
  the same दफा-level convention as item 2).
Same per-document commit/rollback discipline as the rest of that script;
extend its existing `--dry-run` report rather than adding a second flag.

## Governing refs
- `system-design.md` Core Invariant #1, #4, #6, §7.1, §7.7, §9 (integrity
  vs. temporal validity), PS-6.
- `docs/ingestion_design.md` migration-005 header note (chunks = search
  derivative, revalidate against authority).

## Allowed scope
- `migrations/011_chunk_authority_links.sql` (new)
- `scripts/migrate.py`
- `app/ingestion/pipeline.py`
- `app/ingestion/pgvector_indexer.py`
- `app/retrieval/validation_gate.py`
- `scripts/backfill_authority_layer.py`
- `tests/test_ingestion_pipeline.py`, `tests/test_validation_gate.py`,
  `tests/test_backfill_authority_layer.py`

## Forbidden
- Do not touch `app/retrieval/eligibility_gate.py` or
  `app/retrieval/gated_orchestrator.py` — AGENT-19/20's fixes there are
  done and unrelated to this task.
- Do not touch `scripts/ingest_laws.py`, `scripts/review_documents.py`,
  `scripts/backfill_source_kind.py`, `scripts/recompute_content_hashes.py`,
  or `app/authority/writer.py`'s `upsert_source` `kind=` literal —
  AGENT-23's scope, already merged to `dev`; this branch is cut from
  after that merge, so these files are already in their fixed state.
- Do not swap rendered citation text to `expression.text_ne` (see §3
  above) — out of scope, different granularity problem.
- Do not attempt to backfill `not_yet_effective`/commence-side eligibility
  logic — untouched, already correct from AGENT-19/20.

## Required checks
- `make test`
- `make lint`
- `make eval-gates`

## Self-review before returning
- Confirm a claim whose component was repealed *after* its as_of does
  NOT abstain (the per-claim-as-of direction matters as much as the
  abstention direction).
- Confirm `component_uri IS NULL` chunks (nkp_case, or any row not yet
  backfilled) don't crash the new check — they should just skip it, not
  raise.
- Run the extended `backfill_authority_layer.py --dry-run` against
  whatever local corpus state you have and report before/after counts,
  same as prior backfill-script tasks in this repo's history.
- Confirm `_citation()`'s fallback path (NULL `source_pub_id`) still
  returns the same shape as before this task, not a partially-filled dict.

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any
AI tool. No `Co-Authored-By: Claude` trailer, no "Generated with Claude" line.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
