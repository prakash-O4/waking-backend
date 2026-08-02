# Wakil-G — Orchestration Progress

## Current task
None — Phase A complete. Awaiting next task.

## Base branch
`dev`

## Working branch
None

## Status
**IDLE** — Phase A merged.

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

### PA-A — Wire /ask with bitemporal gated pipeline (MERGED, commit d270be2)
- `app/main.py`: Pinecone/Cohere path removed; /ask now uses dumb_retriever →
  model (claims+evidence_ids only) → validate_and_render(); as_of per-request
- `tests/test_ask_pipeline.py`: 4 tests (happy path, no-hits, model abstain, gate abstain)
- `Makefile`: app/main.py added to lint paths
- make lint / make test / make eval-gates all green

## Phase 0 + A status
**COMPLETE.**
- 677-law corpus ingestable (dumb BM25 baseline)
- Eligibility gate + validation gate on every /ask path
- Model never writes citations; server gate owns citation render and abstention
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migration 002 to Supabase

## Governing design refs
- system-design.md §2 (Core Invariants), §13 (Phase A), §14 (PS-1…PS-18)
- AGENTS.md (prime directive, definition of done)

## Next action
Awaiting Prakash's direction. Next milestone: Phase B (orchestrator + resilience)
or any sub-task Prakash prioritises.
