# AGENT-20 — Approved-only eligibility + document approval CLI

## Branch
`agent/document-approval-gate` (base: `dev`, cut after AGENT-19 merged —
`eligibility_gate.py` already has the `nkp_case` exclusion; don't remove it)

## Objective
Three findings from a 2026-09-01 diagnostic pass, all grounded against
`system-design.md` + the live code:

1. **Core Invariant #5 / PS-2 — live violation.** `eligible_chunk_ids()`
   returns `d.ingestion_status IN ('approved', 'pending')`. Unreviewed
   documents are retrievable today. A test,
   `test_eligible_chunk_ids_includes_pending_valid_document`, explicitly
   locks this in — it is not accidental drift, and it must change as part
   of this fix, not be left contradicting the new behavior.
2. **No document-level approval path exists anywhere in the codebase.**
   `scripts/review_lifecycle.py` only approves `lifecycle_effect` rows
   (commence/repeal/amend proposals) — a different table. Nothing has ever
   set `documents.ingestion_status = 'approved'` outside a raw SQL session.
   Without this, fix #1 makes retrieval permanently empty — that's the
   *correct* failure mode (a loud refusal beats a quiet wrong answer), but
   this CLI is what makes the system usable again.
3. **PS-3 mislabel.** `writer.py::upsert_source()` hardcodes
   `kind='official_copy_unverified'` for every `laws.jsonl` source.
   `docs/ingestion_design.md` §1.2 calls this content *"a consolidation,
   not the original gazette text"* produced by a third party (the `content`
   field is sourced from signed GCS URLs under `eksana_legal`/`eksanaai`,
   confirmed by inspecting `laws.jsonl` directly) — per PS-3 it must be a
   derived kind, not an official-copy kind.

## Scope (four items — keep each one's diff separable in your commit even
though they land in one PR)

### 1. `eligible_chunk_ids()` — approved only
Change `d.ingestion_status IN ('approved', 'pending')` to
`d.ingestion_status = 'approved'`.

### 2. `eligible_chunk_ids()` — NULL `effective_date_ad` excluded
Change `c.effective_date_ad IS NULL OR c.effective_date_ad <= %(as_of)s` to
require a non-NULL, past-or-equal date. **Do not** attempt to join live
against `lifecycle_effect` for this — `chunks.effective_date_ad` is already
correctly populated at ingest time from an *approved* `commence` row
(`pipeline.py::_commence_date()`), so a straight NULL exclusion achieves
the correctness goal (nothing with unresolved commencement is ever
eligible) without reimplementing that lookup's URI-construction logic
(Devanagari-digit translation, `{work_uri}/dafa/{section}` format) a second
time in raw SQL, where a subtle mismatch would fail silently. This is a
deliberate narrowing from the original ask ("include only if authority
says commenced") to the safely-implementable version of the same
guarantee — flag if you think the live join is worth the risk instead, but
don't build it without discussing first.
**Known, separate, out-of-scope gap** (do not fix here): the cache can
still drift the *other* direction — a chunk's `effective_date_ad` predates
a later repeal/amendment that hasn't re-triggered a backfill. §7.4 already
names this risk. Out of scope for this task.

### 3. `scripts/review_documents.py` (new)
Dual-approval CLI for the `documents` table, mirroring
`scripts/review_lifecycle.py`'s mechanics (`_approve_one`/`approve_one`
around line 71, `_reject_one`/`reject_one` around line 122) — same
`FOR UPDATE` row lock, same distinct-approver enforcement
(`approved_by == by` on the second call is refused), same two-call
pattern (first call records `approved_by`, second *different* `by` flips
`ingestion_status='approved'` and sets `second_approved_by`). Differences
from `lifecycle_effect`'s shape: `documents.approved_by`/`second_approved_by`
are `TEXT`, not UUID — keep that. `--list` (pending documents),
`--approve <document_id> --by <name>`, `--reject <document_id> --by <name>`.
Do not extend `review_lifecycle.py` itself — different table, different
approver-identity shape, and a shared CLI would make "approve X" ambiguous
about which table it targets.

### 4. `writer.py::upsert_source()` — `derived_verified`
Change the hardcoded `kind='official_copy_unverified'` to `'derived_verified'`.
**Honest scope note, put in your own summary too**: this changes what's
*written*, not what's *displayed* — `validation_gate.py::_citation()`
doesn't read `source_publication.kind` at all today (it reads
`chunks.source_type`, a different column with a different meaning, and
hardcodes `ocr_confidence: None`). That's a separate, deeper PS-10 gap
(there's no FK from `documents`/`chunks` to the specific `source_publication`
row backing them — `pipeline.py` computes `source_pub_id` but never
persists it onto `documents`) that needs a schema decision. Out of scope
here — flagged separately to Prakash, not yours to fix.

## Governing refs
- `system-design.md` Core Invariant #5, PS-2, PS-3, §6 stage-8 pause point.
- `docs/ingestion_design.md` §6 (pipeline stages, dual-approval pause),
  §1.2 (laws.jsonl provenance), migration 005's `documents_dual_approval`
  CHECK constraint (the schema-level backstop your CLI logic must satisfy
  on the first try, not rely on).

## Allowed scope
- `app/retrieval/eligibility_gate.py`
- `tests/test_eligibility_gate.py`
- `scripts/review_documents.py` (new)
- `tests/test_review_documents.py` (new)
- `app/authority/writer.py` (only the `kind=` literal in `upsert_source`)

## Forbidden
- Do not touch `app/retrieval/validation_gate.py`, `gated_orchestrator.py`,
  `pipeline.py`, or any schema/migration file.
- Do not remove or weaken the `nkp_case` exclusion AGENT-19 added.
- Do not implement the live `lifecycle_effect` join described (and
  rejected) in item 2.
- Do not couple document approval to lifecycle_effect approval status in
  code (no gate requiring lifecycle proposals be resolved before a document
  can be approved) — §6 treats this as a human judgment call during
  review, not a code-enforced dependency.

## Required checks
- `make test` — including rewriting/replacing
  `test_eligible_chunk_ids_includes_pending_valid_document` (must now
  assert pending is **excluded**) and the substring assertion in
  `test_eligible_chunk_ids_excludes_quarantined` (SQL text changed).
- `make lint`
- `make eval-gates`

## Self-review before returning
- Confirm the two now-contradictory existing tests were actually updated,
  not left passing-by-coincidence or silently deleted without replacement
  coverage.
- Confirm `scripts/review_documents.py`'s dual-approval logic can't be
  satisfied by one person calling `--approve` twice with the same `--by`.
- Re-read `docs/ingestion_design.md`'s idempotency note
  ("hash different → ...`ingestion_status` reset to `pending`") and confirm
  your CLI doesn't need to handle re-approval differently — a
  content-hash-changed document goes back through the same pending queue,
  no special case needed.

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any
AI tool. No `Co-Authored-By: Claude` trailer, no "Generated with Claude" line.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
