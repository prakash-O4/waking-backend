# task.md — AGENT-26: canonical eligibility predicate, pre-retrieval and at validation

## Objective

An external retrieval-quality review (independently verified by Claude before
this brief was written — see PROGRESS.md's AGENT-26 entry for the full
grounding) found that the eligibility gate and the validation gate implement
**two different, drifting reimplementations** of "is this component currently
good law," and neither of them is the canonical one that already exists.

`migrations/001_bitemporal_schema.sql` + `migrations/002_gate_suspend_fix.sql`
define `is_eligible(component_uri, as_of)` — a correct SQL function checking:
approved `commence` effect covering `as_of` with no pending
`commencement_dependency`, AND no approved `repeal`/`expiry`/
`declared_invalid`/`suspend` effect with `lower(legal_valid_time) <= as_of`.

**Nobody in the live `/ask` path calls it.** It is only called from
`app/eval/gates.py` (the offline zero-tolerance gate scorer, tested against
synthetic insert/rollback data — not against real retrieval traffic). Concretely:
a court `declared_invalid` ruling is unenforceable anywhere in the live query
path today. `app/retrieval/eligibility_gate.py::eligible_chunk_ids()` only
checks `ingestion_status='approved'`, `source_type<>'nkp_case'`, and a
denormalized `effective_date_ad <= as_of` cache column — no repeal/expiry/
suspend/declared_invalid check at all, pre-retrieval. And
`app/retrieval/validation_gate.py::_terminated_before()` is a second,
independent reimplementation that checks `repeal/expiry/suspend` but is
**missing `declared_invalid`** — silently out of sync with the canonical
function.

Fix: make `is_eligible()` (already correct, do not modify it or the
migrations) the single source of truth for temporal/lifecycle eligibility,
and have both `eligible_chunk_ids()` (pre-retrieval) and
`_terminated_before()` (validation-time, defense-in-depth) derive from it
instead of re-deriving their own predicates.

## Acceptance criteria

1. `eligible_chunk_ids()` excludes a chunk whose linked component is
   `repeal`/`expiry`/`suspend`/`declared_invalid`-terminated as of `as_of`,
   or not yet `commence`-effective as of `as_of` (including the
   `commencement_dependency` pending-notification case) — **before**
   retrieval runs, not just at validation. It must keep its existing,
   unrelated checks: `ingestion_status='approved'`, `source_type<>'nkp_case'`.
2. `_terminated_before()` (or its replacement) uses the same predicate as
   `eligible_chunk_ids()` — no second copy of the repeal/expiry/suspend/
   declared_invalid condition list living in a different function with a
   different set of effect_types. One source, two call sites, not two sources.
3. A component under an approved `declared_invalid` effect is excluded both
   pre-retrieval and at validation. A component under an approved `suspend`
   effect is excluded during the suspension window (Prakash's explicit call —
   a suspended provision is not currently good law even though it may be
   reinstated later).
4. **Required grounding step before writing the join**: query the live
   corpus for `chunks.component_uri` population coverage broken down by
   `chunk_type`/`source_type` (the AGENT-24/25 history in PROGRESS.md notes
   `subsection` chunks are deliberately unlinked by design — no दफा-level
   parent row is ever created for them — while `proviso`/`tariff_row`/
   `tariff_note` are ~100% linked). A chunk with no `component_uri` cannot be
   checked against `lifecycle_effect` at all. Report the actual coverage
   numbers in your completion report. **Do not silently decide** whether an
   unlinked chunk should default to eligible or ineligible if the coverage
   data shows this would flip a large, legitimate population's retrievability
   — stop and flag it back with the numbers instead of guessing. If coverage
   confirms only the already-known-unlinked-by-design `subsection` case is
   affected, keep those retrievable via the existing `effective_date_ad`
   fallback (documented invariant, not a gap) and apply the new check to
   every chunk that does carry a `component_uri`.
5. No behavior change to anything else `eligible_chunk_ids()` or
   `_terminated_before()` currently does (approved-only, non-`nkp_case`,
   span-hash check, citation rendering) — this task is scoped to the
   temporal/lifecycle predicate only.
6. Existing tests in `tests/test_eligibility_gate.py` that hardcode
   assertions against the *current* weak SQL (e.g.
   `test_eligible_chunk_ids_excludes_future_effective_date` asserting
   `"c.effective_date_ad IS NOT NULL"` is present) encode the old
   implementation, not a frozen contract — update them to match the new
   query shape rather than preserving the old SQL text. Reuse the existing
   `FilteringCursor`/`FilteringConn` pattern (actually evaluates predicates
   against seeded rows) for the new coverage — do not write string-only SQL
   assertions for new tests.
7. New tests must prove, with seeded data (not canned booleans): a
   `declared_invalid` component is excluded pre-retrieval; a `suspend`ed
   component is excluded during its suspension window but becomes eligible
   again once the suspension's `legal_valid_time` upper bound passes (if the
   schema/effect model supports a bounded suspension — check
   `lifecycle_effect.legal_valid_time`'s shape before assuming); a
   not-yet-`commence`d component (or one with a pending
   `commencement_dependency`) is excluded pre-retrieval, not just via the
   `effective_date_ad` cache. Mirror the `TerminationCursor`/`TerminationConn`
   real-predicate-evaluation style already in `tests/test_validation_gate.py`.

## Branch

`agent/canonical-eligibility-gate` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: eligibility gate + validation gate on every
  path; every claim validates against its declared as-of.
- `system-design.md` §2 Core Invariant #1 (bitemporal store is single
  authority; derivatives are revalidated) and #2 (deterministic eligibility
  gate runs pre-retrieval, on every branch, same predicate every path).
- `system-design.md` §14: **PS-2** (commencement-pending must never be
  silently in-force), **PS-4** (`declared_invalid` court events must be
  enforced, not just stored), **PS-15** (status enum distinguishes repealed/
  spent/lapsed — extend the reasoning to suspend/declared_invalid here).

## Allowed scope

- `app/retrieval/eligibility_gate.py`
- `app/retrieval/validation_gate.py` — **only** `_terminated_before()` and
  its call site inside `validate_and_render()`. Do **not** touch `_citation()`
  or `_expression()` — citation-chain rendering is a separate task
  (AGENT-28), already queued.
- `tests/test_eligibility_gate.py`, `tests/test_validation_gate.py`
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not modify `migrations/001_bitemporal_schema.sql` or
  `migrations/002_gate_suspend_fix.sql` — `is_eligible()` is already correct
  for this task's scope. If you find it actually needs a change, stop and
  report why rather than editing it.
- Do not touch `_citation()`, `gated_orchestrator.py`, `query_graph.py`,
  `postgres_retriever.py`, or anything precedent-related
  (`precedent_retriever.py`, `phase_d_slice.py`) — all out of scope for this
  task, each is a separate queued task or explicitly deferred.
- Do not add a new dependency, service, table, or abstraction. This is a
  predicate-unification fix inside existing functions.
- Do not change jurisdiction or ACL filtering — out of scope, not requested.

## Required checks

- `make test`
- `make lint`
- `make eval-gates` — `repealed-as-current` and `not-yet-effective-as-current`
  must both stay at `0`. (`overruled-as-good-law` is precedent-only, untouched
  by this task, must also stay `0` simply because nothing here should affect
  it.)

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

## Self-review before returning

- No retrieval or validation path skips either gate.
- The model still only ever sees claims/evidence the *new* predicate allows —
  confirm by tracing that `eligible_chunk_ids()`'s result set is what
  actually gates `postgres_retriever.py`'s vector/lexical queries (it already
  does today via the `eligible` list — just confirm your change doesn't
  break that wiring, don't change `postgres_retriever.py` itself).
- Report the live-corpus `component_uri` coverage numbers from step 4 even
  if they show no surprises.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
