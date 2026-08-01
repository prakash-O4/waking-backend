---
name: ps-check
description: Production Safety Annex compliance check for Wakil-G. Run before touching any ingestion, gate, temporal, precedent, or calendar path. Maps affected code to PS-1…PS-18 and verifies each requirement has a test.
---

# PS-Check — Production Safety Annex Compliance

Run this skill before implementing (or reviewing) any change that touches:
- ingestion pipeline (Side A)
- eligibility gate or validation gate (pre- or post-retrieval)
- temporal logic (as-of resolution, valid_time, transaction_time, commencement)
- precedent subsystem (bench competence, holding-level status, relation validity)
- BS↔AD calendar conversion
- abstention or citation rendering
- authority store schema or bitemporal writes
- eval or stress harness gates

## Step 1 — Identify affected PS requirements

Read the diff or task description. For each changed component, mark which PS requirements it touches:

| PS | Requirement summary | Trigger |
|---|---|---|
| PS-1 | Precedent status at holding level, competent bench, as-of derived | Any precedent or `precedent_relation` change |
| PS-2 | `राजपत्रमा सूचना` → `not_yet_effective` + commencement_dependency, no fabricated date | Commencement logic, lifecycle_effect parsing |
| PS-3 | Citations → authoritative instrument chain; consolidation labeled `derived`; CI reconstructs expression | Citation rendering, expression materialization, ingestion publish |
| PS-4 | Store norm level, competence list, enabling_power link, `declared_invalid`; disclose `in_force` ≠ validity | Work/component schema, norm-hierarchy logic |
| PS-5 | BS↔AD versioned canonical data; boundary windows warn; future-range dates abstain | Calendar conversion, as-of resolution |
| PS-6 | As-of is per-claim; each claim validates against its declared as-of | Claim assembly, validation gate, multi-hop orchestrator |
| PS-7 | Abstention is server-owned; model self-abstention is advisory only | Validation gate, abstention logic, model output handling |
| PS-8 | Romanized Nepali is a first-class eval slice with its own numeric target | Eval harness, language ID, query understanding |
| PS-9 | Translation divergence checked (back-translation/NLI), routed to reviewers, surfaced to user; not auto-corrected | Translation pipeline, multilingual path |
| PS-10 | OCR/source-kind provenance as citation reliability badge renders to user | Citation renderer, source_publication provenance |
| PS-11 | Postgres-down = no validated answers; replica is synchronous or freshness-gated vs lifecycle-write watermark | Degraded-mode ladder, replica logic |
| PS-12 | Stress gate = append-only suite no-regression + fresh red-team per release; coverage by taxonomy cells | Stress harness, release gate |
| PS-13 | LLM-judge κ ≥ 0.6, per-slice and rolling; below-threshold → 100% human review | Eval harness, judge calibration |
| PS-14 | Traces store IDs/hashes by default; raw content in separate short-retention dual-control store; redact at collector | Observability, OTel config, trace schema |
| PS-15 | Status enum distinguishes repealed / spent / lapsed | Status enum, lifecycle_effect logic |
| PS-16 | Provisos and स्पष्टीकरण co-retrieve with operative clause (eval-asserted) | Chunking, retrieval, context assembly |
| PS-17 | Retroactive amendments: close prior valid_time at retroactive date; flag affected historical answers; regression test straddles both axes | Bitemporal write logic, retroactive amendment handling |
| PS-18 | Stillborn statute and code-switched romanized+English query bucket are explicitly tested | Eval/stress coverage, query understanding edge cases |

## Step 2 — For each affected PS, verify

For every PS requirement marked above, answer:

1. **Does the change maintain this requirement?** (Yes / No / Partial — explain)
2. **Is there an existing test covering this requirement?** (Yes — name it / No — must be written)
3. **If no test exists, write one before this merges.** A change that touches a PS-guarded path without a test is not done.

## Step 3 — Zero-tolerance gate check

If the change touches statute temporal logic, citation rendering, or the eligibility/validation gates, explicitly confirm:
- `repealed-as-current = 0` still enforced
- `not-yet-effective-as-current = 0` still enforced
- `overruled-as-good-law = 0` still enforced (Phase D+ only; wired but unenforced before)

Run `make eval-gates` and confirm all three are green.

## Step 4 — Core Invariant check (§2)

Confirm the change does not weaken any of the 10 Core Invariants:
1. Bitemporal store is the single authority — no derivative promoted to source of truth
2. Eligibility gate on every retrieval branch — no new path skips it
3. Model emits `claims + evidence_ids` only — no citation writing by the model
4. Server-side validation gate on every answer path — no bypass
5. Ingestion is human-gated, dual approval — no programmatic promotion of `pending` items
6. Every claim validates against its declared as-of — no session-level as-of shortcuts
7. Abstention is server-owned — model self-abstention is advisory only
8. All retrieved text is untrusted — no raw statutory text injected unguarded
9. Statute and precedent are separate lifecycle models — no cross-contamination
10. Provenance propagates to user — no silent dropping of OCR confidence or source kind

## Output Format

```
PS-Check — [task/diff description]

Affected PS requirements: [list]

| PS | Maintained? | Test exists? | Action needed |
|---|---|---|---|
| PS-X | Yes/No/Partial | Yes (test_name) / No | [write test / fix / none] |
...

Zero-tolerance gates:
- repealed-as-current = 0: [CONFIRMED / AT RISK — reason]
- not-yet-effective-as-current = 0: [CONFIRMED / AT RISK — reason]
- overruled-as-good-law = 0: [N/A (pre-Phase-D) / CONFIRMED / AT RISK — reason]

Core Invariants: [ALL MAINTAINED / INVARIANT N WEAKENED — describe]

Decision: [CLEAR TO PROCEED | BLOCKED — list what must be resolved first]
```
