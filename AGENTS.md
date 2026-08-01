# AGENTS.md — Wakil-G working contract

Wakil-G answers Nepali legal questions bound to a point in time, with citations that resolve to authoritative instruments. This file is the working agreement for any coding agent in this repo. It is **not** the architecture — that lives in `SYSTEM_DESIGN.md`, and the enforced requirements live in its Production Safety Annex (`PS-1…PS-18`). Read those before changing anything in the ingestion, gate, temporal, or precedent paths.

## Prime directive
**A loud refusal beats a quiet wrong answer.** When unsure, abstain. Never make the system more confident than its evidence.

## Non-negotiables (do not violate; each maps to a PS-requirement)
- The bitemporal store is the only authority. Indexes/embeddings/consolidations are derivatives.
- The model emits `claims + evidence_ids` only. It never writes citations and never mutates state.
- The eligibility gate (pre-retrieval) and the validation gate (server-side) are on every path. Don't add a code path that skips them.
- Every claim validates against its declared as-of.
- Ingestion of legal state is human-gated with dual approval. No exceptions in code.
- All retrieved text is untrusted input.
If a change would weaken one of these, stop and raise it — it's an architecture decision (ADR), not a patch.

## How to work here (Karpathy skills)
Adapted from *A Recipe for Training Neural Networks* and applied to this system.

1. **Become one with the data.** Before writing a parser, chunker, or extractor, read the raw Gazette / statute / नजिर text by hand. The corpus teaches the edge cases; don't guess them.
2. **Skeleton first, dumbest baseline that runs.** Wire the whole path with a trivial retriever and no reranker before adding anything clever. A dumb baseline that runs beats a smart design that doesn't.
3. **Overfit one case, then widen.** Make one statute, one as-of query, one citation perfect end-to-end before generalizing.
4. **Keep the generate→verify loop tight.** Fast local evals, run constantly. Verification is the bottleneck — if evals are slow, everything is slow.
5. **Don't be a hero.** No bespoke complexity where a boring, tested, deterministic component works. Simple and correct wins.
6. **Treat the LLM as a jagged intern.** Brilliant, confidently wrong, no memory. Never trust — verify. (This is *why* the model never writes citations.)
7. **Autonomy slider.** Keep a human on the loop for lifecycle writes and high-risk answer classes. Turn autonomy up only where evals prove it safe.
8. **Measure, don't guess.** "It feels better" is not a reason. A change ships because a numeric slice moved, or it doesn't ship.

## Definition of done
- Touches a fixed-invariant path → the relevant `PS-*` test is written/updated and green.
- Zero-tolerance eval gates stay at zero: `repealed-as-current`, `not-yet-effective-as-current`, `overruled-as-good-law`.
- Stress suite: no regression + red-team round if you added a language path, tool, or index version.
- Change is small, reviewed, ADR-linked if it moves a threshold/prompt/index, and passes full regression before redeploy (canary, rollback ready).

## Commands
<!-- fill in for the chosen stack; keep this the single source for agents -->
- Setup: `make setup`
- Test: `make test`
- Evals: `make eval` (slice report) · `make eval-gates` (zero-tolerance)
- Stress: `make stress`
- Lint/format: `make lint`

## When to stop and ask a human
High-risk answer classes, anything touching lifecycle writes, any change that would alter a gate predicate or a temporal rule, and any case where evidence is thin enough that the honest move is to abstain.