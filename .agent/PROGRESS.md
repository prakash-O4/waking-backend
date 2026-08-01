# Wakil-G — Orchestration Progress

## Current task
P0-A — Infrastructure skeleton (Makefile, bitemporal schema, OpenSearch setup)

## Base branch
`dev`

## Working branch
`phase-0/infra-skeleton`

## Engineer
Pi

## Status
**IN PROGRESS** — task.md written and pushed to branch. Awaiting Pi's implementation.

## Repository state (verified 2026-08-01)

**Phase:** Pre-Phase 0 prototype. A working RAG prototype exists but the System Design architecture is not yet in place.

**What exists:**
- FastAPI backend (`app/main.py`) — GPT-4o-mini via LangChain, Pinecone for vector search, Cohere reranker, Supabase for auth/chat history
- Ingestion pipeline (`app/ingestion/`) — document_processor, hierarchical_chunker, metadata_enricher, pinecone_indexer, quality_validator
- Retrieval pipeline (`app/retrieval/`) — advanced_retriever, query_processor, retrieval_orchestrator
- One statute indexed: `sarbajanik` (Sarbajanik something)
- Domain detection: out-of-domain queries return empty sources
- LangSmith tracing enabled

**Critical gaps vs. System Design:**
1. No PostgreSQL bitemporal authority store — Pinecone is the only store (violates Invariant 1)
2. No deterministic eligibility gate (temporal validity, status, jurisdiction) — Invariant 2
3. Model currently writes citations directly from context — violates Invariant 3
4. No server-side validation gate — Invariant 4
5. No human-gated dual-approval ingestion workflow — Invariant 5
6. No per-claim as-of validation — Invariant 6
7. No BS↔AD calendar data or handling
8. No `make` harness: `make test`, `make eval`, `make eval-gates`, `make stress`, `make lint` do not exist
9. No eval/stress layer — zero-tolerance gates untested
10. Governance files (AGENTS.md, CLAUDE.md, system-design.md) are untracked — not yet committed

**No active working branches.** All code is on `dev`.

## Governing design refs
- SYSTEM_DESIGN.md §2 (Core Invariants), §14 (PS-1…PS-18)
- AGENTS.md (prime directive, definition of done)

## PS requirements in scope
None (schema only — no ingestion or retrieval in P0-A)

## Zero-tolerance gates
Not applicable for P0-A (no data flows yet)

## Next action
Prakash runs Pi on branch `phase-0/infra-skeleton`. Pi returns commit hash + check outputs. Claude reviews diff.
