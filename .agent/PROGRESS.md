# Wakil-G — Orchestration Progress

## Current task
None — Phase 0 complete. Awaiting next task.

## Base branch
`dev`

## Working branch
None

## Status
**IDLE** — Phase 0 merged.

## Completed tasks

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal Supabase schema, OpenSearch client, Pydantic models

### P0-B — Full corpus ingestion + dumb baseline (MERGED, commit 9f0f086 + beeb05bf + 9b08516)
- `app/authority/parser.py`: parses laws.jsonl → components, BS→AD dates, strips amend tags
- `app/authority/writer.py`: idempotent bitemporal writes
- `app/retrieval/eligibility_gate.py`: wraps is_eligible() SQL
- `app/retrieval/dumb_retriever.py`: BM25 + per-hit eligibility filter
- `app/retrieval/validation_gate.py`: span hash verify + citation render (Invariants 3+4)
- `app/eval/gates.py`: zero-tolerance gate tests — both pass
- `scripts/ingest_laws.py`: ingests 677 laws from laws.jsonl
- `scripts/query.py`: end-to-end CLI demo
- migration 002: eligibility gate suspend fix
- loguru added to requirements.txt (pre-existing gap)

**Review findings resolved:**
- Pi blocker: `app/retrieval/__init__.py` clobbered → restored by Pi
- loguru missing dep → fixed by Claude during merge

## Phase 0 status
**COMPLETE.** Schema migrated, 677-law corpus ingestable, dumb BM25 baseline
with eligibility + validation gates wired, eval-gates green.

**Still pending before Phase A:**
- Run `scripts/ingest_laws.py` against real Supabase + OpenSearch (requires env vars)
- Run `scripts/migrate.py` to apply migration 002 to Supabase
- Wire new pipeline into `app/main.py` `/ask` endpoint (Phase A)
- Replace Phase 0 stub dual-approval with real human-gating UI (Phase A)
- BS↔AD canonical calendar table (Phase C)

## Governing design refs
- SYSTEM_DESIGN.md §2 (Core Invariants), §14 (PS-1…PS-18)
- AGENTS.md (prime directive, definition of done)

## Next action
Awaiting Prakash's direction. Next milestone: Phase A (statute path — both gates
fully wired into /ask, per-claim as-of, zero-tolerance eval gates green in prod).
