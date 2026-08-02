# Wakil-G — Orchestration Progress

## Current task
**PA-A** — Wire `/ask` with bitemporal gated pipeline (Phase A)

## Base branch
`dev`

## Working branch
`phase-a/statute-path`

## Status
**IN PROGRESS** — task brief written, awaiting Pi.

## Owner
Pi

## Governing design refs
- system-design.md §2 (Core Invariants), §13 (Phase A), §14 (PS-3, PS-6, PS-7)
- AGENTS.md (prime directive, definition of done)

## PS requirements in scope
- PS-3: Citations from `source_publication + work` metadata only, via `validate_and_render()`
- PS-6: `as_of` per-request, passed to every retrieve + validate call
- PS-7: Model abstention advisory; server validation gate owns abstention

## Zero-tolerance gates (must stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

## Next action
Prakash runs Pi on `phase-a/statute-path` with the prompt below.
Pi implements, tests, commits. Returns result to Claude for review.

---

## Completed tasks

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal Supabase schema, OpenSearch client, Pydantic models

### P0-B — Full corpus ingestion + dumb baseline (MERGED, commit 9f0f086 + beeb05bf + 9b08516)
- `app/authority/parser.py`: parses laws.jsonl → components, BS→AD dates, strips amend tags
- `app/authority/writer.py`: idempotent bitemporal writes
- `app/retrieval/eligibility_gate.py`: wraps is_eligible() SQL
- `app/retrieval/dumb_retriever.py`: BM25 + per-hit eligibility filter
- `app/retrieval/validation_gate.py`: span hash verify + citation render (Invariants 3+4)
- `app/eval/gates.py`: zero-tolerance gate tests — both pass
- `scripts/ingest_laws.py`: ingests 677 laws from laws.jsonl
- `scripts/query.py`: end-to-end CLI demo
- migration 002: eligibility gate suspend fix
- loguru added to requirements.txt (pre-existing gap)

## Phase 0 status
**COMPLETE.** Schema migrated, 677-law corpus ingestable, dumb BM25 baseline
with eligibility + validation gates wired, eval-gates green.

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migration 002 to Supabase
