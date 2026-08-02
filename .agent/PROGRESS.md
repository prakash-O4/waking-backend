# Wakil-G — Orchestration Progress

## Current task
**PE-A — Ingestion Pipeline Research & Design**
Stack migration: Supabase + Pinecone → plain PostgreSQL + pgvector.
Redesign chunking (structure-aware for Nepali legal text), metadata extraction,
PII redaction, hybrid RDB+vector schema, and pipeline stages.
Deliverable: `docs/ingestion_design.md` (design only, no implementation).

## Base branch
`dev`

## Working branch
`pe-a/ingestion-pipeline`

## Status
**IN PROGRESS** — task.md written, branch pushed. Assigned to Kimi.

## PS requirements in scope
- PS-2: dual approval gate (ingestion_status enum)
- PS-3: citations to authoritative chain (not chunks)
- PS-14: PII redaction (pii_vault table)
- PS-16: provisos co-retrieve with operative clause (chunking constraint)

## Open architectural question (Prakash must decide before implementation)
BM25 gap: pgvector alone drops BM25 Nepali retrieval (required by §8).
Options: PostgreSQL tsvector / keep OpenSearch / pg_search (ParadeDB).
Kimi will research and recommend; Prakash decides.

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

### PD-A — Precedent schema + eval gate + retriever skeleton (MERGED, commit 96533eb)
- `migrations/004_precedent_schema.sql`: `precedent`, `precedent_holding`, `precedent_relation` tables + `is_good_law(uuid, date)` SQL function
- `app/authority/precedent_models.py`: `RelationType` enum, `PrecedentRelation` dataclass
- `app/retrieval/precedent_retriever.py`: ILIKE over holdings with `is_good_law()` filter
- `app/eval/gates.py`: `check_overruled_as_good_law()` + updated `main()`
- `scripts/migrate.py`: applies migration 004
- `tests/test_precedent_gate.py`: 5 mock-based tests
- 5 new tests (29 total passing, 1 skipped)

## Phase 0 + A + B + C + D status
**COMPLETE.**
- All gates enforced on every path (including all degraded modes)
- bs_to_ad_approx() eliminated; canonical calendar live (PS-5)
- Romanized Nepali is a first-class eval slice with Recall@5 (PS-8)
- Precedent subsystem: holding-level model, bench-competence gate (PS-1)
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0, overruled-as-good-law = 0

## Operational steps still pending (on Prakash)
- Run `scripts/migrate.py` (applies migrations 001-004)
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch
- Run `scripts/seed_bs_ad_calendar.py` after migration 003
- Run `make eval` against live env to get baseline Recall@5
- Run `make eval-gates` against live env to verify all gates against real DB
- Ingest precedent corpus (then wire `retrieve_precedent` into orchestrator)

## Governing design refs
- system-design.md §2, §6, §7.6, §10, §13, §14 (PS-1, PS-5, PS-8)
- AGENTS.md (prime directive, definition of done)

## Next action
Awaiting Prakash's direction. All four build phases complete.
