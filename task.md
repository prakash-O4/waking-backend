# AGENT-19 — Hard-disable NKP/precedent retrieval until Phase D

## Branch
`agent/nkp-precedent-lockout` (base: `dev`)

## Objective
`system-design.md` §6 is explicit: *"Case law does not ship until this subsystem
exists (Phase D). Until then, Supreme Court is ingested but not answered
from."* Today it is answered from anyway: `eligibility_gate.py::eligible_chunk_ids()`
returns chunks of any `source_type`, including `nkp_case`, and
`gated_orchestrator.py::_authority_rank_hits()` tiers `nkp_case` hits (tier 6)
instead of excluding them — there is no `overruled-as-good-law` check anywhere
in the query path (confirmed by grep: `validation_gate.py` has zero
precedent/overrule-awareness). This is a live violation of §6 and PS-1, not a
"not built yet" gap — the design never granted case law a grace period before
its lifecycle model exists.

Fix: exclude `nkp_case` chunks at the eligibility gate — the single choke
point every retrieval path already funnels through (`postgres_retriever.py`
and `gated_orchestrator.py::_resolve_cross_refs` both call
`eligible_chunk_ids()`; confirmed no other code path queries `chunks` directly).
`app/retrieval/precedent_retriever.py` (the real Phase D path, gated by
`is_good_law()`) is separate, already correctly designed, and only called from
`app/eval/phase_d_slice.py` today — out of scope, do not touch it.

## Acceptance criteria
- `eligible_chunk_ids()` never returns a chunk with `source_type = 'nkp_case'`,
  regardless of `ingestion_status` or `effective_date_ad`.
- `act`/`regulation` chunk eligibility behavior is unchanged — this is a pure
  exclusion, not a rewrite of the existing predicate.
- New test: an `nkp_case` chunk on an `approved` document with a past
  `effective_date_ad` (i.e. would pass every other filter) must NOT appear in
  `eligible_chunk_ids()`'s output.
- Existing tests in `tests/test_eligibility_gate.py` continue to pass or are
  updated only where they assert the exact SQL string (the `'pending'` clause
  itself is explicitly **out of scope** — see Forbidden below).

## Governing refs
- `system-design.md` §6 (precedent subsystem, ships Phase D only), §8 (query
  plane — eligibility gate is the pre-retrieval choke point on every branch),
  Core Invariant #2 (deterministic eligibility gate, same predicate on every
  path), Core Invariant #9 (statute and precedent are separate lifecycle
  models — one does not serve the other early).
- PS-1: precedent status must be derived at holding level from
  competent-bench relations; until that model is wired, case law must not be
  answered from at all.

## Allowed scope
- `app/retrieval/eligibility_gate.py` — add the `source_type` exclusion.
- `tests/test_eligibility_gate.py` — new/updated tests for the exclusion.

## Forbidden
- Do **not** touch the `ingestion_status IN ('approved', 'pending')` clause
  or the `'pending'`-inclusion tests — that is a separate task (AGENT-20,
  queued after this one) and touching it here will conflict.
- Do **not** modify `gated_orchestrator.py`. Its `nkp_case` tiering branch
  (line ~363) becomes naturally dead code once no `nkp_case` chunk ever
  reaches it — leave it as-is; removing dead code is not this task's scope.
- Do **not** touch `app/retrieval/precedent_retriever.py`,
  `app/eval/phase_d_slice.py`, or anything under `app/authority/precedent_models.py`.
- No schema/migration changes — this is a query-predicate change only.

## Required checks
- `make test`
- `make lint` (`app/retrieval/eligibility_gate.py` is already in the fixed
  lint file list — no manual ruff/mypy carve-out needed)
- `make eval-gates` (not expected to move — none of the three zero-tolerance
  gates touch `nkp_case`/`source_type` today; confirm unchanged, don't just
  assume)

## Self-review before returning
- Re-grep for any retrieval path that queries `chunks` without going through
  `eligible_chunk_ids()` — confirm the claim above ("single choke point") is
  actually still true on your diff, not just trusted from this brief.
- Confirm no path renders an `nkp_case` citation end-to-end (a quick
  synthetic-data test through `validation_gate.py` if one doesn't already
  exist cheaply).

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any
AI tool. No `Co-Authored-By: Claude` trailer, no "Generated with Claude" line.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
