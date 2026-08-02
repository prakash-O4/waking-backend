# Wakil-G — Orchestration Progress

## Current task
None — Phase B complete. Awaiting next task.

## Base branch
`dev`

## Working branch
None

## Status
**IDLE** — Phase B merged.

---

## Completed tasks

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal Supabase schema, OpenSearch client, Pydantic models

### P0-B — Full corpus ingestion + dumb baseline (MERGED, commit 9f0f086 + beeb05bf + 9b08516)
- `app/authority/parser.py`, `writer.py`, `eligibility_gate.py`, `dumb_retriever.py`,
  `validation_gate.py`, `eval/gates.py`, `scripts/ingest_laws.py`, `scripts/query.py`
- migration 002: eligibility gate suspend fix

### PA-A — Wire /ask with bitemporal gated pipeline (MERGED, commit d270be2)
- `app/main.py`: eligibility gate → dumb_retriever → model (claims+evidence_ids) → validate_and_render
- `tests/test_ask_pipeline.py`: 4 tests

### PB-A — Gated orchestrator + degraded-mode ladder (MERGED, commit 5dfbfcc)
- `app/retrieval/gated_orchestrator.py`: LLM router (simple/complex), multi-hop up to 3
  sub-queries each with own as_of, wall-clock cap 20s, all paths gate-validated
- `app/retrieval/postgres_retriever.py`: ILIKE fallback retriever with eligibility gate
- `app/main.py`: delegates to orchestrator_answer(); Postgres-down → HTTP 503 (PS-11)
- `tests/test_degraded_modes.py` + `tests/test_orchestrator.py`: 10 new tests
- make lint / make test (15 passed) / make eval-gates all green

## Phase 0 + A + B status
**COMPLETE.**
- Eligibility gate + validation gate on every path (including all degraded modes)
- Model never writes citations; server gate owns citation render and abstention
- Per-claim as_of enforced on every hop
- Postgres-down → 503, no index-only fallback
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migration 002 to Supabase

## Governing design refs
- system-design.md §2 (Core Invariants), §8, §9, §13, §14 (PS-6, PS-7, PS-11)
- AGENTS.md (prime directive, definition of done)

## Next action
Awaiting Prakash's direction. Next milestone: Phase C (language hardening —
Romanized Nepali slice, BS↔AD canonical calendar) or Phase D (precedent).
