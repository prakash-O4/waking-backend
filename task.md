# AGENT-21 — Stop calling pending documents "processed" in ingest_laws.py

## Branch
`agent/ingest-status-messaging` (base: `dev`, no overlap with AGENT-19/20 —
different file, can run independently of those)

## Objective
`scripts/ingest_laws.py::main()` buckets every document that reaches
`DUAL_APPROVAL_PAUSE` (i.e. `document_id is not None`) into
`counts["processed"]`, printed as `processed={N}` in the final summary.
The pipeline never sets `ingestion_status` to anything but `pending` at
this point (`pipeline.py`'s own module docstring: *"The pipeline never sets
ingestion_status to 'approved' — that flip is human-only via a separate
admin path"*) — calling it "processed" reads as done/successful when it
actually means "queued, awaiting the dual-approval CLI." Purely a
messaging/naming fix — no behavior change.

## Scope
- `scripts/ingest_laws.py`: rename the `counts["processed"]` key and its
  corresponding `f"  ✓ {outcome}"` / final summary label to something that
  doesn't imply completion — e.g. `pending_review` (matches
  `documents.ingestion_status`'s actual value at this point). Update both
  the per-record `print` and the final `"done: processed=... "` summary
  line consistently. Don't change the counting logic itself (which
  document_ids land in this bucket) — only the label.

## Allowed scope
- `scripts/ingest_laws.py` only.

## Forbidden
- No change to `IngestionPipeline`/`pipeline.py` logic, no change to what
  counts as skipped/rejected/failed — this is a label rename only.
- No new tests required (no existing test file covers this script's CLI
  output — don't add test infrastructure for a messaging-only change).

## Required checks
- `make lint` (this file is in the fixed lint list).
- `make test` (confirm nothing else references the old `"processed"` key —
  grep before you rename).

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any
AI tool. No `Co-Authored-By: Claude` trailer, no "Generated with Claude" line.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
