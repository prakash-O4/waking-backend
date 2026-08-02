# Wakil-G — Orchestration Progress

## Current task
**PB-A** — Orchestrator + degraded-mode ladder (Phase B)

## Base branch
`dev`

## Working branch
`phase-b/orchestrator-resilience`

## Status
**IN PROGRESS** — task brief written, awaiting Pi.

## Owner
Pi

## Governing design refs
- system-design.md §2 (Core Invariants), §8 (Query plane), §9 (Degraded-mode ladder), §13 (Phase B), §14 (PS-6, PS-7, PS-11)
- AGENTS.md (prime directive, definition of done)

## PS requirements in scope
- PS-6: Per-claim as_of — each result carries the as_of used to validate it
- PS-7: Model abstention advisory; server gate owns abstention on every path
- PS-11: Postgres-down → HTTP 503, no fallback to index-only answers

## Zero-tolerance gates (must stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

## Next action
Prakash runs Pi on `phase-b/orchestrator-resilience`. Returns result to Claude for review.

---

## Completed tasks

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal Supabase schema, OpenSearch client, Pydantic models

### P0-B — Full corpus ingestion + dumb baseline (MERGED, commit 9f0f086 + beeb05bf + 9b08516)
- `app/authority/parser.py`, `writer.py`, `eligibility_gate.py`, `dumb_retriever.py`,
  `validation_gate.py`, `eval/gates.py`, `scripts/ingest_laws.py`, `scripts/query.py`
- migration 002: eligibility gate suspend fix

### PA-A — Wire /ask with bitemporal gated pipeline (MERGED, commit d270be2)
- `app/main.py`: Pinecone/Cohere removed; eligibility gate → dumb_retriever →
  model (claims+evidence_ids) → validate_and_render; as_of per-request
- `tests/test_ask_pipeline.py`: 4 tests
- make lint / make test / make eval-gates all green

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migration 002 to Supabase
