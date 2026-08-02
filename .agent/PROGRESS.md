# Wakil-G — Orchestration Progress

## Current task
**PC-A** — BS↔AD canonical calendar + Romanized Nepali eval slice (Phase C)

## Base branch
`dev`

## Working branch
`phase-c/language-hardening`

## Status
**IN PROGRESS** — task brief written, awaiting Pi.

## Owner
Pi

## Governing design refs
- system-design.md §2 (Core Invariants), §7.6 (BS↔AD as canonical data), §10 (eval), §13 (Phase C), §14 (PS-5, PS-8)
- AGENTS.md (prime directive, definition of done)

## PS requirements in scope
- PS-5: BS↔AD is versioned canonical data; BeyondCalendarRange for out-of-range dates
- PS-8: Romanized Nepali is a first-class eval slice with its own Recall@5 metric

## Zero-tolerance gates (must stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

## Next action
Prakash runs Pi on `phase-c/language-hardening`. Returns result to Claude for review.

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
- `app/retrieval/gated_orchestrator.py`, `app/retrieval/postgres_retriever.py`
- Degraded modes: Postgres-down→503, OpenSearch-down→Postgres fallback, model-down→extractive
- 10 new tests; make lint / make test (15 passed) / make eval-gates all green

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migrations 002 + 003 to Supabase
- Run `scripts/seed_bs_ad_calendar.py` after migration 003 is applied
