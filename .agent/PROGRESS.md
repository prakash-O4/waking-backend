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
**DESIGN APPROVED** — `docs/ingestion_design.md` reviewed and approved (commit ab4dcb4).
Three open questions require Prakash's decision before implementation begins (see below).

## PS requirements in scope — all verified GREEN
- PS-2: dual-approval CHECK constraint in DDL; effective_date_ad NULL-pending (never fabricated)
- PS-3: laws.jsonl content treated as derived_verified (consolidation); <amend> tags preserved
- PS-5: BS→AD only via bs_ad_calendar; boundary-window dates flagged for human review
- PS-10: ocr_confidence column on documents; regulations OCR provenance explicit
- PS-14: pii_vault with REVOKE ALL + pii_vault_reader role; retrieval path never sees raw PII
- PS-16: co_retrieve_parent_id FK enforces proviso co-retrieval; eval-asserted

## Implementation note (minor, not a blocker)
co_retrieve_parent_id is a self-referential FK on chunks. UPSERT stage must insert
parent chunks before child (proviso) chunks within each document — chunk_index order
guarantees this, but the implementation engineer must not batch-insert out of order.

## Open questions — Prakash decides before implementation

1. **BM25 / text-search**: Kimi recommends pg_search (ParadeDB) — same transaction boundary,
   no OpenSearch. All options lack Nepali stemming, so morphology is better at query time.
   Fallback: PostgreSQL tsvector simple config if ParadeDB unavailable on host.
   → **Prakash: confirm Option C (pg_search) or override.**

2. **Regulations scope**: regulations.json has no text — only PDF URLs. Ingesting regulations
   requires a fetch + PDF→markdown + parse step (Azure DI quota + PDF availability to confirm).
   → **Prakash: is regulation ingestion in PE-B scope or later?**

3. **Embedding model**: bge-m3 (1024-dim) recommended; bake-off on golden eval slice before
   full embed run. DDL is already sized at vector(1024) for both bge-m3 and e5-large.
   → **Prakash: confirm bge-m3 or request bake-off first.**

## Next action
Prakash decides the three open questions above.
Once decided: implementation engineer (Kimi, same branch) writes migration 005 and
the ingestion pipeline code.

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
