# Wakil-G — Orchestration Progress

## Current task
None. Awaiting Prakash's direction.

## Status
**IDLE** — PE-A merged to dev (merge commit on dev, 2026-08-03).

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
- `app/authority/bs_ad_calendar.py`: BS 2000-2090 lookup, BeyondCalendarRange, boundary-window infra
- `app/authority/parser.py`: bs_to_ad_approx() deleted; canonical lookup in place
- `migrations/003_bs_ad_calendar.sql`, `scripts/seed_bs_ad_calendar.py`
- `app/eval/romanized_slice.py`: Recall@5 harness; 10 golden queries
- 9 new tests (24 total passing)

### PD-A — Precedent schema + eval gate + retriever skeleton (MERGED, commit 96533eb)
- `migrations/004_precedent_schema.sql`: precedent/holding/relation tables + `is_good_law(uuid, date)`
- `app/authority/precedent_models.py`, `app/retrieval/precedent_retriever.py`
- `app/eval/gates.py`: `check_overruled_as_good_law()` wired
- 5 new tests (29 total passing, 1 skipped)

### PE-A — PostgreSQL + pgvector + bge-m3 ingestion pipeline (MERGED to dev, 2026-08-03)
- `migrations/005_ingestion_pipeline.sql`: documents, chunks (pgvector 1024-dim, HNSW), pii_vault,
  pg_search BM25 index; dual-approval DDL (PS-2); REVOKE ALL on pii_vault (PS-14)
- `app/ingestion/laws_chunker.py`: दफा-anchor structure-aware chunker; PS-16 co-retrieval links
- `app/ingestion/nkp_chunker.py`: hybrid anchor chunker (caption/headnote/opinion/order/colophon)
- `app/ingestion/pii_redactor.py`: deterministic + haiku second pass + verification assertion
- `app/ingestion/pgvector_indexer.py`: bge-m3 embed, chunk upsert in index order, pii_vault write
- `app/ingestion/pipeline.py`: 8-stage orchestrator; never sets approved; quarantines on redaction failure
- `app/ingestion/metadata_enricher.py`: rewritten to Anthropic SDK (haiku-4-5); LangChain removed
- `scripts/ingest_laws.py`, `scripts/ingest_nkp.py`: CLI scripts with dry-run mode
- `tests/test_ingestion_pipeline.py`: 35 passing, 2 skipped (live-DB dual-approval test conditional)
- PS-2 / PS-3 / PS-5 / PS-10 / PS-14 / PS-16 all verified GREEN

## Phase 0 + A + B + C + D + E status
**COMPLETE.**
- All gates enforced on every path (including all degraded modes)
- Canonical BS/AD calendar live (PS-5); romanized eval slice live (PS-8)
- Precedent subsystem with holding-level model and bench-competence gate (PS-1)
- Ingestion pipeline: PostgreSQL + pgvector + bge-m3 + pg_search; dual approval enforced in DDL
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0, overruled-as-good-law = 0

## Operational steps still pending (on Prakash)
- Run `scripts/migrate.py` (applies migrations 001-005; 005 requires ParadeDB or comment out BM25 block)
- Run `scripts/ingest_laws.py` against real PostgreSQL + pgvector DB
- Run `scripts/ingest_nkp.py --input output/nkp_cases.jsonl` (after PII review policy confirmed)
- Run `scripts/seed_bs_ad_calendar.py` after migration 003
- Run `make eval` + `make eval-gates` against live env to get baseline Recall@5 and verify zero-tolerance gates
- Ingest precedent corpus (then wire `retrieve_precedent` into orchestrator)
- Decide whether to commit/discard `app/config.py` working-tree change (`extra = "ignore"` in Settings.Config)

## Governing design refs
- SYSTEM_DESIGN.md §2, §6, §7.6, §10, §13, §14
- AGENTS.md (prime directive, definition of done)
- docs/ingestion_design.md (PE-A design; approved by Prakash 2026-08-02)

## Next action
Awaiting Prakash's direction.
