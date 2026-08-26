# Wakil-G — Orchestration Progress

## Current task
**AGENT-7 — Unified Langfuse trace: one tree per query**

## Status
**IN PROGRESS** — Branch `agent/stage-7-unified-trace` created. Task brief in `task.md`. Assigned to Pi.

- Base: `dev`
- Branch: `agent/stage-7-unified-trace`
- Engineer: Pi
- PS in scope: PS-14 (LOG_CONTENT flag unchanged)
- Zero-tolerance gates: none touched

### What this fixes
Root cause of "wtf logs": every query produces 3+ disconnected top-level traces with `endTime: null`.
Fix: one `rag.query` root trace per query; retrieval, reasoning, validation, composition all as children;
`lf_trace` threaded via `config["configurable"]` (not QueryState); single `_lf.flush()` in `run_query`.

### Files in scope
- `postgres_retriever.py` — export `get_lf_client`; `retrieve_postgres(lf_trace=None)`; child spans under `retrieval_span` not root trace
- `gated_orchestrator.py` — `_langfuse_callback(trace_id=None)`; `lf_trace=None` on all 3 LLM fns; remove `_emit_answer_trace`; rewrite `_emit_answer_trace_from_state` to update+end existing trace
- `query_graph.py` — create root trace in `run_query`; pass via config; node-level spans; flush at end
- `tests/test_orchestrator.py` — update `test_emit_trace_uses_vector_score` for new signature

### Next action
Prakash runs Pi on branch `agent/stage-7-unified-trace` with `task.md`.

---

## Architecture decisions (standing)

### Authority weighting — Option A: Tier-first, RRF-second (decided 2026-08-24)
Sort retrieved chunks by work_type tier, break ties by RRF score. No blended weights.
Tier order: Constitution(1) > Act(2) > Rule/Regulation(3) > Directive/Byelaw(4) > Notification/Order(5) > Precedent(6).
Rationale: legally deterministic, auditable, no eval data needed to calibrate.
Upgrade path: move to weighted blend (Option B) once eval slice data justifies a specific α/β split.
Ref: `docs/adr-001-multi-agent-query-architecture.md` §Authority Weighting.

### Missing facts handling — Hybrid (decided 2026-08-24)
Fact extractor classifies each missing fact as: required | clarifying | informational.
- required    → interrupt graph, ask user before retrieving
- clarifying  → ask user if within wall-clock budget, else proceed and document
- informational → document in answer output, never blocks
Ref: `docs/adr-001-multi-agent-query-architecture.md` §Missing Facts.

---

---

## Completed tasks

### AGENT-6 — Observability Fix (MERGED to dev, 2026-08-25)
- `postgres_retriever.py`: `_span` → `_end_span` — calls `span.end()` so all spans have `endTime`; `_hit()` gains `vector_score` param; `vector_scores` dict built from vector search rows; cosine similarity propagated through RRF and rerank to returned hits
- `gated_orchestrator.py`: `_langfuse_callback()` now used on all 3 LLM calls (`_structured_claims`, `_fact_extract`, `_compose_answer`); explicit `callbacks[0].langfuse.flush()` after each invoke; `_emit_answer_trace_from_state` uses `vector_score` (not RRF score) for `top_chunk_scores`; `LANGFUSE_LOG_CONTENT` flag gates raw `query` + `answer_summary` fields in trace; `_compose_answer` strips markdown code fences before `json.loads`
- `config.py`: `LANGFUSE_LOG_CONTENT: bool = False` added
- `tests/test_orchestrator.py`: `FakeResp.content` in compose test now uses markdown-wrapped JSON to verify fence stripping; `test_emit_trace_uses_vector_score` added; 67 total passing
- PS-14 maintained: raw content off by default; latent risk noted — `_fact_extract` JSON parse lacks fence stripping (can add in cleanup pass)

### AGENT-5 — Answer Composer + Missing-Facts Interrupt (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_classify_and_decompose` removed (dead since AGENT-2); `import os` removed; `_compose_answer(facts, missing_facts, all_results, conflict_hits, session_as_of)` added — Gemini 2.5 Flash composes ADR Node 7 format (`relevant_sections`, `plain_language`, `missing_facts`, `conflicts`, `disclaimer`); filters to clarifying/informational missing facts only; `except Exception: return None` fallback
- `query_graph.py`: `fact_extractor_node` detects `required` missing facts → sets `interrupted=True` + `interrupt_prompt`; `assemble_node` replaced by `answer_composer_node` — interrupt short-circuit (returns directly, bypasses retrieval) or normal path (Gemini compose + fallback to raw claims); `build_graph()` uses `add_conditional_edges` from `fact_extractor` → `answer_composer` (interrupt) or `retrieve` (normal); graph still 7 nodes, one of which is now reached via two paths
- `tests/test_orchestrator.py`: `test_classifier_failure_falls_back_to_simple` removed; 3 new tests added (compose success, no-key fallback, interrupt integration with retrieve_called == [] assertion); 66 total passing
- Note: `_compose_answer` does not wire Langfuse callbacks into the Gemini call (minor observability gap, consistent with `_fact_extract` pattern — can add in OBS pass)

### AGENT-4 — Reasoner Rewrite (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_model_claims` removed; `_structured_claims(facts, issue_queries, ranked_hits)` added — Azure `gpt-4.1-mini` via `AzureChatOpenAI`, tier-labelled context (32k char cap), structured output with `issue`/`applicability`/`condition`; `_CONTEXT_CHAR_LIMIT` + `_TIER_LABELS` constants added; `azure_base_url` imported; co-retrieved chunks inherit `_issue_idx` from parent hit
- `query_graph.py`: `retrieve_generate_node` split → `retrieve_node` (pure retrieval, `_issue_idx` tagging) + `reasoner_node` (per-issue `_structured_claims` call over authority-ranked context, grouping by `_issue_idx`); `validate_node` updated to propagate `issue`/`applicability`/`condition` from original claims to rendered results; graph now 7 nodes
- `tests/test_orchestrator.py`: tests 1–3 and 5 updated to mock `_structured_claims`; 2 new tests for `_structured_claims` success path and no-key fallback; 64 total passing
- `tests/test_degraded_modes.py`: stale `_model_claims` mock updated to `_structured_claims` (Pi found this proactively)
- PS-6, PS-7, PS-12 verified; zero-tolerance gates at 0

### AGENT-3 — Authority Ranker + Cross-Reference Resolver (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_authority_rank_hits(hits, conn)` — queries `chunks.work_id → work.work_type` (LEFT JOIN), sorts by `(tier ASC, score DESC)`, attaches `tier` + `conflict_flag` (same section_number, lower tier); `_resolve_cross_refs(hits, as_of, conn)` — regex scans top-10 hits for `दफा/उपदफा/अनुसूची X`, fetches eligible co-chunks via `eligible_chunk_ids`, appends with `co_retrieved=True`; both have `except Exception` top-level guard; `_REAL_MONOTONIC` removed; `_WORK_TYPE_TIER`, `_DEVA_DIGIT_MAP`, `_CROSS_REF_RE` constants added; `eligible_chunk_ids` re-exported for mock compatibility
- `query_graph.py`: `authority_ranker_node` and `cross_ref_resolver_node` inserted between `retrieve_generate` and `validate`; graph now 6 nodes
- `tests/test_orchestrator.py`: 5 new tests; 62 total passing
- Schema note: ADR says `documents.work_type` but correct path is `chunks.work_id → work.work_type`; conflict detection is document-agnostic (same section_number across different works can trigger it — acceptable for Stage 3; Stage 4/5 can scope by work if needed)

### AGENT-2 — Fact Extractor + issue-driven retrieval (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_fact_extract()` added — Gemini 2.5 Flash extracts `facts`, `missing_facts`, and `issue_queries` (Devanagari Nepali queries, max 3); fallback to single raw query on any failure or missing `GEMINI_API_KEY`
- `query_graph.py`: `classify_node` replaced by `fact_extractor_node`; `retrieve_generate_node` iterates `issue_queries` instead of `subqueries`; `_graph_clock` removed (was CPython-specific frame-walking); `run_query` now uses `_orch.time.monotonic()` for mock-compatible wall_clock_start
- `tests/test_orchestrator.py`: tests 1–3 and 5 updated to mock `_fact_extract`; test 4 unchanged; 2 new tests for `_fact_extract` success path and fallback; 57 total passing
- Cleanup note: `_REAL_MONOTONIC` in `gated_orchestrator.py` (line 19) is now unused — remove in Stage 3 sweep

### AGENT-1 — LangGraph skeleton (MERGED to dev, 2026-08-24)
- `query_state.py`: `QueryState` TypedDict — full schema incl. Stage 2+ placeholders
- `query_graph.py`: 4-node linear graph (classify → retrieve_generate → validate → assemble); `conn` via `config["configurable"]`; all calls via `_orch.*` for monkeypatch compatibility
- `gated_orchestrator.py`: `answer()` delegates to `run_query()`; all helpers remain at module level; `retrieve_postgres` + `validate_and_render` re-exported; `_emit_answer_trace_from_state` extracted
- `requirements.txt`: `langgraph>=1.2`
- 55 tests passing, no behaviour change
- Cleanup note: `_graph_clock` in `query_graph.py` uses `sys._getframe` (CPython-specific, solves non-existent problem in LangGraph 1.2 sync path) — remove in next cleanup cycle

### RET-C — FlashRank fallback reranker (MERGED to dev, 2026-08-24)
- `reranker.py`: full rewrite — Cohere → FlashRank (`ms-marco-MultiBERT-L-12`, multilingual) → passthrough ladder; module-level `_ranker` cache; `except Exception: pass` on Cohere falls through silently
- `requirements.txt`: `flashrank` added
- `tests/test_retrieval.py`: old `test_reranker_skipped_when_cohere_key_unset` replaced with 4 tests covering full ladder; 53 total passing
- No PS requirements in scope; no gates affected (post-retrieval path)

---

## Completed tasks

### RET-B — Dual-path cross-lingual query translation (MERGED to dev, 2026-08-24)
- `postgres_retriever.py`: `_is_devanagari()` (U+0900–U+097F, 0.5 threshold); `translate_query()` via Gemini 2.5 Flash (`langchain-google-genai`); dual-path vector + lexical search when translation succeeds; 4-list RRF fusion; `translation_ran` in eligibility_gate span
- `config.py`: `GEMINI_API_KEY: str = ""`
- `requirements.txt`: `langchain-google-genai>=2.0`
- `tests/test_retrieval.py`: 8 new tests (50 total passing); Cursor mock upgraded to SQL-content detection for dual-path correctness
- Graceful degradation: translation failure → single-path fallback, no exception
- PS-8 served (Romanized Nepali eval slice); no invariants weakened; lint clean

---

## Completed tasks

### OBS-RET — Retrieval observability + eval slice (MERGED to dev, 2026-08-23)
- `postgres_retriever.py`: 6 Langfuse stage spans (eligibility → vector → lexical → RRF → relevance gate → rerank); each with latency_ms, counts, scores
- `gated_orchestrator.py`: answer trace expanded — retrieval/generation/validation latency, claims_passed/abstained, top_chunk_scores
- `romanized_slice.py`: fixed URI matching (source_id based, not URI prefix)
- `retrieval_slice.py`: new — Recall@1/3/5 + MRR; baseline 0.4 / MRR 0.33 on 200 laws
- `eligibility_gate.py`: dropped `valid_time` transaction-time check (was blocking all retrospective queries)
- PS-14 compliant; 42 tests passing

### RET-A — Retrieval layer rewrite (MERGED to dev, 2026-08-23)
- `postgres_retriever.py`: full rewrite — preprocessing, Azure query embedding,
  eligibility gate, vector ANN + tsvector GIN, RRF fusion, relevance gate, Cohere rerank
- `eligibility_gate.py`: new `eligible_chunk_ids()` querying `documents`/`chunks` (not old `lifecycle_effect`)
- `reranker.py`: new Cohere wrapper, opt-in (no-op if `COHERE_API_KEY` unset)
- `validation_gate.py`: resolves evidence_ids against `chunks` table (not `expression`)
- `migrations/007_retrieval_indexes.sql`: GIN tsvector index on `chunks.chunk_text`
- `config.py`: added `COHERE_API_KEY: str = ""`
- PS-6, PS-7, PS-12 verified; 42 tests passing, 2 skipped

### CLEANUP-A — Remove OpenSearch (MERGED to dev, 2026-08-22)
- Deleted: `app/search/client.py`, `app/search/__init__.py`, `app/retrieval/dumb_retriever.py`, `docker-compose.yml`
- `gated_orchestrator.py`: removed `_try_retrieve`, direct `retrieve_postgres` call
- `requirements.txt`: removed `opensearch-py==2.7.1`
- `Makefile`: removed OpenSearch startup from `setup` target
- `tests/test_degraded_modes.py`: deleted 2 OS tests, fixed 1 mock
- Eval slices + `scripts/query.py`: swapped to `retrieve_postgres` + `connect()`
- 33 passed, 2 skipped, 0 failed (count drop = 2 deleted OS tests that were passing)

### PH-OBS-B — Full stage-level ingestion tracing (MERGED to dev, 2026-08-22)
- Replaced terminal-only `_emit_ingestion_span` with per-document traces
- Module-level Langfuse singleton — one client for entire ingestion run
- One trace per document (`ingestion.law` / `ingestion.nkp_case`)
- One timed child span per stage: LOAD, VALIDATE, CHUNK, EXTRACT_METADATA,
  EMBED_AND_UPSERT, DUAL_APPROVAL_PAUSE (+ REDACT_PII for NKP)
- Each span carries: stage name, outcome, latency_ms
- `ImportError` guard — degrades to no-op if langfuse package not installed
- PS-14 compliant; 35 tests passing

### PH-OBS-A — Langfuse RAG tracing integration (MERGED to dev, 2026-08-22)
- `langfuse>=2.0` added to `requirements.txt`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `config.py`
- `LangfuseCallbackHandler` wired into LangChain LLM calls in `gated_orchestrator.py`
- `answer()` emits trace: `query_hash` (SHA-256 only), `as_of`, `query_type`,
  `latency_ms` (full elapsed), `retrieved_uris` (component_uri only), `gate_decision`, `result_count`
- `IngestionPipeline` emits per-stage spans: `source_id`, `source_type`, `stage`, `outcome` — no raw content
- PS-14 compliant: no raw query text or statutory text in any span
- Opt-in: no-op when `LANGFUSE_PUBLIC_KEY` unset
- 35 tests passing, lint clean

### PG-B — Azure OpenAI + laws ingestion (2026-08-07, on dev)
- Switched embeddings: `BAAI/bge-m3` (local SentenceTransformer) → Azure OpenAI `text-embedding-3-large`
  - `dimensions=1024` preserves existing schema; no migration required
  - `DEFAULT_BATCH_SIZE` raised from 32 → 512 (no GPU memory constraint with API)
- Switched LLM: standard OpenAI → Azure OpenAI `gpt-4.1-mini`
  - `AzureChatOpenAI` in `metadata_enricher.py`; `AzureOpenAI/AsyncAzureOpenAI` in `ragas_eval.py`
  - Separate `AZURE_OPENAI_LLM_KEY` + `AZURE_OPENAI_LLM_ENDPOINT` fields (distinct from embedding)
  - `azure_base_url()` helper in `config.py` strips deployment path from full endpoint URL
- `sentence-transformers` removed from `requirements.txt`
- Ingestion optimisations:
  - Chunk-metadata LLM call batched at 20 chunks/call (was unbounded — caused 2+ min hangs on large acts)
  - Batches parallelised via `ThreadPoolExecutor(max_workers=3)` — ~2.5× speedup on large acts
  - 60s timeout on `AzureChatOpenAI` (was SDK default 600s — caused silent 10-min hangs)
  - OOM retry loop removed from `_embed()` — irrelevant for API calls
- `scripts/ingest_laws.py`: per-record progress printed to stdout (`[N/total] source_id … ✓ ingested`)
- **100 laws ingested** to local Postgres: 4,263 chunks, Nepali summaries + keywords generated
- Est. cost for 100 laws: ~$1.10 (LLM ~$0.90 + embeddings ~$0.20)

### PG-A — RAGAS v0.2 eval slices per pipeline phase (MERGED to dev, 2026-08-06)
- `ragas==0.2.*` added to `requirements.txt`
- `app/eval/ragas_eval.py`: `BaseRagasLLM` + `BaseRagasEmbeddings` via `openai.AsyncOpenAI`
  directly — no LangchainLLMWrapper, no langchain-openai version conflict
- `app/eval/__init__.py`: minimal shim for `langchain_community.chat_models.vertexai`
  (removed in langchain-community 0.4.x; stub lets ragas 0.2.* import cleanly)
- Phase slices: phase_a (Faithfulness + ResponseRelevancy), phase_c (ContextRecall +
  NonLLMContextPrecisionWithReference), phase_d (stubbed — skips if precedent empty),
  phase_ef (summary Faithfulness vs. source chunks)
- `app/eval/metrics/temporal_faithfulness.py`: custom PS-6-aligned LLM-judge metric
- Golden sets: `phase_a_qa.json` (10), `phase_c_romanized.json` (10), `phase_d_precedent.json` (5 placeholders)
- `make eval`: runs all slices + romanized Recall@5; `make eval-gates` unchanged
- PS-6, PS-13 in scope; zero-tolerance gates all at 0

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal schema, OpenSearch client, Pydantic models

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
- `app/ingestion/pii_redactor.py`: deterministic + LLM second pass + verification assertion
- `app/ingestion/pgvector_indexer.py`: bge-m3 embed, chunk upsert in index order, pii_vault write
- `app/ingestion/pipeline.py`: 8-stage orchestrator; never sets approved; quarantines on redaction failure
- `scripts/ingest_laws.py`, `scripts/ingest_nkp.py`: CLI scripts with dry-run mode
- `tests/test_ingestion_pipeline.py`: 35 passing, 2 skipped
- PS-2 / PS-3 / PS-5 / PS-10 / PS-14 / PS-16 all verified GREEN

### PE-A/fix — Provider-agnostic LLM via LangChain 1.3.0 (MERGED to dev, 2026-08-03)
- `metadata_enricher.py`, `pii_redactor.py`: `anthropic` SDK replaced with `init_chat_model(settings.LLM_MODEL)`
- `config.py`: `LLM_MODEL: str = "openai:gpt-4o-mini"` — swap provider via env var, no code change
- `requirements.txt`: langchain==1.3.14, langchain-openai==1.4.1, langchain-community==0.4.2; anthropic removed
- Collateral: `langchain.schema.Document` → `langchain_core.documents.Document` (removed in LangChain 1.x)

### PF-A — Local Postgres setup + summary field (2026-08-06, on dev)
- **Local Postgres**: Docker container `wakilg-postgres` (pgvector/pgvector:pg17, port 5433)
  - All 13 tables created; pgvector extension live; 33,238 BS/AD calendar rows seeded
  - `DATABASE_URL=postgresql://wakilg:wakilg@localhost:5433/wakilg` in `.env`
- **`app/authority/writer.py`**: reads `DATABASE_URL` first, falls back to `SUPABASE_DB_URL`
- **`migrations/006_add_summary.sql`**: `ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT`
- **`app/ingestion/metadata_enricher.py`**: summary added to both enrichment paths
  - NKP: extracted in same first LLM call as `cited_statutes` + `headnotes` (no extra API call)
  - Laws: one extra LLM call per act (act name + first 5 chunks → 2-3 sentence Nepali summary)
- **`app/ingestion/pipeline.py`**: extracts `summary` from enricher output, passes to document dict
- **`app/ingestion/pgvector_indexer.py`**: writes `summary` into documents upsert
- **`scripts/migrate.py`**: fully rewritten — idempotent via `schema_migrations` tracking table;
  detects pre-existing migrations by object probes; BM25 block auto-skipped on standard Postgres
- Dry-run verified: 1022 NKP cases in `output/nkp_cases.jsonl`, 5/5 sample valid, 0 rejected

## Phase 0 + A + B + C + D + E + F status
**Schema and pipeline COMPLETE. Corpus ready. Ingestion not yet run.**
- All gates enforced on every path (including all degraded modes)
- Canonical BS/AD calendar live (PS-5); romanized eval slice live (PS-8)
- Precedent subsystem with holding-level model and bench-competence gate (PS-1)
- Ingestion pipeline: PostgreSQL + pgvector + bge-m3; dual approval enforced in DDL
- Documents carry: keywords, relevant_questions, cited_statutes, headnotes, **summary**
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0, overruled-as-good-law = 0

## Operational steps still pending (on Prakash)
- Run `scripts/ingest_laws.py` for remaining 577 laws (100 done, 677 total)
- Run `scripts/ingest_nkp.py --input output/nkp_cases.jsonl` (1022 NKP cases)
- Run `make eval` + `make eval-gates` against live env (baseline Recall@5 + zero-tolerance gate check)
- Ingest precedent corpus (then wire `retrieve_precedent` into orchestrator)
- Rewrite `app/main.py` auth layer (Supabase auth → new architecture; `app/utils/helpers.py` SupabaseHelper to be replaced)
- Push `dev` to origin when ready

## Architecture notes
- **DB**: Self-hosted PostgreSQL on VPS (Docker locally). No Supabase dependency for ingestion or retrieval.
  `app/main.py` still has Supabase auth — that is old architecture, to be replaced.
- **LLM**: Azure OpenAI `gpt-4.1-mini` via `AzureChatOpenAI`. Keys: `AZURE_OPENAI_LLM_KEY` + `AZURE_OPENAI_LLM_ENDPOINT`.
- **Embeddings**: Azure OpenAI `text-embedding-3-large` at `dimensions=1024`. Keys: `AZURE_OPENAI_KEY` + `AZURE_OPENAI_ENDPOINT`.
- **BM25**: pg_search (ParadeDB) not available on standard Postgres — falls back to GIN tsvector at query time.

## Governing design refs
- AGENTS.md (prime directive, definition of done)
- docs/ingestion_design.md (PE-A design; approved by Prakash 2026-08-02)

## Next action
Awaiting Prakash's direction.
