# Wakil-G — Orchestration Progress

## Current task
None — awaiting Prakash's direction.

## Base branch
`dev` (up to date with `origin/dev`)

## Working branch
None

## Status
**IDLE** — Session resume complete. No active task.

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

## Next action
Awaiting Prakash's task assignment.
