# Wakil-G — Orchestration Progress

## Current task
None — Phase C complete. Awaiting next task.

## Base branch
`dev`

## Working branch
None

## Status
**IDLE** — Phase C merged.

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
- `tests/test_ask_pipeline.py`

### PB-A — Gated orchestrator + degraded-mode ladder (MERGED, commit 5dfbfcc)
- `app/retrieval/gated_orchestrator.py`, `app/retrieval/postgres_retriever.py`
- Degraded modes: Postgres-down→503, OpenSearch-down→Postgres fallback, model-down→extractive
- 10 new tests

### PC-A — Canonical BS/AD calendar + romanized eval slice (MERGED, commit 4b74709)
- `app/authority/bs_ad_calendar.py`: embedded month-lengths BS 2000-2090, lookup(),
  BeyondCalendarRange, _BOUNDARY_WINDOWS infra
- `app/authority/parser.py`: bs_to_ad_approx() deleted; canonical lookup in place
- `migrations/003_bs_ad_calendar.sql`, `scripts/seed_bs_ad_calendar.py`
- `app/eval/romanized_slice.py`: Recall@5 harness
- `app/eval/golden/romanized.json`: 10 romanized queries with real corpus URIs
- 9 new tests (24 total passing)

## Phase 0 + A + B + C status
**COMPLETE.**
- All gates enforced on every path (including all degraded modes)
- bs_to_ad_approx() eliminated; canonical calendar live (PS-5)
- Romanized Nepali is a first-class eval slice with Recall@5 (PS-8)
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0

## Operational steps still pending (on Prakash)
- Run `scripts/migrate.py` (applies migrations 001-003)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch
- Run `scripts/seed_bs_ad_calendar.py` after migration 003
- Run `make eval` against live env to get baseline Recall@5

## Governing design refs
- system-design.md §2, §7.6, §10, §13, §14 (PS-5, PS-8)
- AGENTS.md (prime directive, definition of done)

## Next action
Awaiting Prakash's direction. Next milestone: Phase D (precedent subsystem —
holding-level model, bench-competence gate, overruled-as-good-law = 0).
