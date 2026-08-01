# Wakil-G — Orchestration Progress

## Current task
P0-B — One statute ingested end-to-end, dumb retrieval chain, eval gates running

## Base branch
`dev`

## Working branch
`phase-0/end-to-end`

## Engineer
Pi

## Status
**IN PROGRESS** — task.md written, pushed to phase-0/end-to-end. Awaiting Pi's implementation.

## Completed tasks

### P0-A — Infrastructure skeleton (MERGED to dev, commit 553a4e3 + merge)
- Makefile: setup / test / lint / eval / eval-gates / stress
- migrations/001_bitemporal_schema.sql: 5 bitemporal tables + is_eligible() SQL gate
- app/authority/models.py: Pydantic v2 models for all authority entities
- app/search/client.py: OpenSearch client + index creation
- docker-compose.yml: local OpenSearch 2.13
- tests/: pytest harness, model + schema fixtures

**Known gaps to fix in P0-B:**
- `app/search/client.py:46` — nori_tokenizer (Korean) should be icu_tokenizer/standard
- `app/authority/models.py:19` — ComponentType missing bhag (भाग), khanda (खण्ड)
- `migrations/001_bitemporal_schema.sql:86` — eligibility gate: suspend not excluded

## Governing design refs
- SYSTEM_DESIGN.md §2 (Core Invariants), §14 (PS-1…PS-18)
- AGENTS.md (prime directive, definition of done)

## PS requirements in scope (P0-B)
- PS-2: commencement pending → not_yet_effective (ingestion path)
- PS-3: citations → authoritative chain; expression labeled derived
- PS-6: as-of per-claim (retrieval path)
- PS-7: abstention server-owned (validation gate stub)
- PS-15: status enum distinguishes repealed/spent/lapsed

## Zero-tolerance gates (P0-B)
- repealed-as-current = 0: must be enforced by is_eligible() + test
- not-yet-effective-as-current = 0: commencement_dependency must block retrieval

## Next action
Prakash runs Pi on branch `phase-0/end-to-end`. Pi returns commit hash + check outputs + query demo. Claude reviews diff.
