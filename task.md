# task.md — AGENT-27: claim-support verbatim-quote check

## Objective

The server-side validation gate (`app/retrieval/validation_gate.py::validate_and_render`)
currently checks span-hash integrity + eligibility + termination, but never
verifies that a claim's text is actually supported by its cited chunk. A
model can emit a plausible-sounding claim tied to an `evidence_id` whose
`chunk_text` says something else entirely, and the gate has no way to catch
it today. This is the biggest confirmed gap from the retrieval-quality review
(see `.agent/PROGRESS.md`'s "Retrieval-quality review + 6-task program" entry,
finding #3).

Design decision already made by Prakash, not open for reinterpretation: the
claim-support check is a **deterministic verbatim-quote substring match**,
not an NLI/semantic-entailment model. The model emits a short verbatim quote
copied from its context chunk alongside each claim; the server checks that
quote is an actual (normalized) substring of the chunk's stored `chunk_text`.
No new dependency, no new service call.

This implements `system-design.md` §7.7's "exact-quote check → claim-support
check" as a single deterministic step (Prakash's simplification of the two
listed steps into one mechanism — do not build them as two separate checks).

## Acceptance criteria

1. **Prompt change** (`app/retrieval/gated_orchestrator.py::_structured_claims`):
   the JSON schema the model is asked to emit gains a required `"quote"`
   field per claim — instruct the model explicitly to copy a short, exact,
   contiguous substring from the CONTEXT chunk it cites (not a paraphrase,
   not a translation, not assembled from multiple places in the chunk) that
   supports the claim. Keep the rest of the schema and prompt structure
   unchanged.
2. **Extractive fallback** (`app/retrieval/gated_orchestrator.py::_extractive_claim`):
   set `"quote"` equal to the same `hit["text_ne"][:EXTRACTIVE_CHARS]` value
   already used for `"claim"` — this path is definitionally verbatim
   (`text_ne` **is** `chunks.chunk_text`, confirmed at
   `postgres_retriever.py:122`), so it always self-supports. Do not add a
   database round-trip here.
3. **Server-side check** (`app/retrieval/validation_gate.py::validate_and_render`):
   add a claim-support check that runs alongside the existing span-hash /
   eligibility / termination checks (same `ok = ok and ...` chain — if it
   fails, the claim abstains exactly like the other failure modes do today,
   same `citation: None` shape, no new response field required).
   - Compare the claim's `quote` against `chunk_text` (the same `expr[0]`
     already fetched by `_expression()` — do not add a second query).
   - Normalize both sides before comparing: NFC + Devanagari-digit-fold +
     collapse-whitespace-and-strip. Reuse the existing normalization pattern
     already duplicated in `app/ingestion/pipeline.py::_content_hash` and
     `app/ingestion/pii_redactor.py::_digit_fold` (NFC normalize +
     `str.maketrans` digit-fold table) — do not import across the
     ingestion/retrieval boundary; a third small local copy in
     `validation_gate.py` matches the codebase's existing precedent of
     duplicating this exact two-line pattern rather than introducing a
     shared-utils module for it (Ponytail: smallest local change, not a new
     abstraction).
   - A missing, empty, or whitespace-only `quote` fails the check (does not
     crash — `claim.get("quote", "")`).
   - Guard against a degenerate short-string match: after normalization and
     stripping, a `quote` shorter than **15 characters** fails the check
     regardless of whether it happens to substring-match. (Chosen by Claude
     as the orchestrating architect to close a real gaming vector — e.g. a
     single common word trivially "supporting" any claim — without a second
     round-trip to ask; this is a tunable parameter, not a proven-optimal
     one, and may be revisited once eval data exists per `AGENTS.md`'s
     "measure, don't guess." Flag in your report if 15 looks wrong against
     real corpus quote lengths you observe while testing.)
4. **Claim shape propagation** (`app/retrieval/query_graph.py::validate_node`):
   the per-claim field copy-through loop (currently copies `issue`/
   `applicability`/`condition` from `orig_claims[i]` onto the validated
   `result`) must also copy `"quote"` through, so a claim's supporting quote
   is visible on the final validated/rendered result for tracing and future
   debugging — not just consumed internally and discarded.
5. No change to what `_citation()`, `_terminated_before()`, or
   `eligible_chunk_ids()` do — this task adds one new check to the existing
   chain, it does not touch the others (AGENT-26 already landed the
   canonical predicate wiring; AGENT-28 owns citation-chain rendering next —
   do not touch `_citation()`).
6. Existing tests in `tests/test_validation_gate.py` construct claims as
   `{"claim": "x", "evidence_id": "/c/1"}` with no `"quote"` — update the
   ones that exercise the full `validate_and_render` success path to include
   a `"quote"` that is a genuine substring of the mocked `_expression()` text
   (so they keep passing for the reason they're supposed to pass, not by
   accident of the check being skipped). Tests that exercise a specific
   earlier failure mode (bad hash, terminated component) may keep omitting
   `quote` since the chain should already short-circuit before reaching the
   new check — but add an explicit assertion that verifies this is actually
   short-circuiting, not silently passing due to the new check being lenient.
7. New tests must prove, with real behavior (not canned booleans):
   - A claim whose quote is an exact substring of `chunk_text` passes.
   - A claim whose quote is *not* found in `chunk_text` at all abstains
     (`citation is None`), even though span-hash/eligibility/termination all
     pass.
   - A claim whose quote differs from the chunk only in digit script
     (Devanagari vs. ASCII) or whitespace/newline layout still passes (the
     normalization is doing real work, not decorative).
   - A claim with an empty or missing `quote` abstains.
   - A claim with a quote shorter than the length guard abstains even when
     it technically substring-matches.

## Branch

`agent/claim-support-verbatim-quote` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: server-side validation gate is on every
  path; abstention is server-owned; retrieved text is untrusted (the quote
  itself is model output over untrusted context — treat it as an untrusted
  string to be checked, not as a citation source; citations still only ever
  come from `_citation()`'s canonical-metadata resolution, never from the
  claim or the quote).
- `system-design.md` Core Invariant #4 (server-side validation gate resolves
  evidence, verifies span hash, verifies temporal validity, **checks claim
  support**, then renders) and #7 (abstention is server-owned; model
  self-abstention is advisory only).
- `system-design.md` §7.7: `... → exact-quote check → claim-support check →
  bounded regenerate (max N) → then fallback/abstain → render citations`.
  Note: **bounded regenerate is explicitly out of scope for this task** — do
  not implement a retry/regenerate loop. On claim-support failure, abstain
  directly (same as every other failure mode in the current chain). Bounded
  regenerate, if ever built, is a separate future task.
- `system-design.md` §14 **PS-7** (abstention is server-owned).

## Allowed scope

- `app/retrieval/validation_gate.py`
- `app/retrieval/gated_orchestrator.py` — **only** `_structured_claims`'s
  prompt/schema and `_extractive_claim`. Do not touch `_compose_answer` or
  anything else in this file (AGENT-29 owns composer re-validation next).
- `app/retrieval/query_graph.py` — **only** the field copy-through loop
  inside `validate_node`. Do not touch anything else in this file.
- `tests/test_validation_gate.py`
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not add an NLI model, embedding-similarity check, or any new
  dependency/service call for claim support. Deterministic substring match
  only, per Prakash's explicit design decision.
- Do not implement bounded regenerate (see §7.7 note above).
- Do not touch `_citation()`, `eligible_chunk_ids()`, `_terminated_before()`,
  `postgres_retriever.py`, `_compose_answer`, or anything precedent-related.
- Do not add a shared normalization utils module — duplicate the small
  NFC+digit-fold pattern locally, matching existing codebase precedent.

## Required checks

- `make test`
- `make lint`
- `make eval-gates` — `repealed-as-current` and `not-yet-effective-as-current`
  must stay at `0` (untouched by this task, must show no regression).
  `overruled-as-good-law` likewise untouched, must also stay `0`.

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

(This task does not touch the eligibility/termination predicates that guard
these gates — they must show zero regression, not zero by virtue of being
untouched-and-unverified. Run `make eval-gates` and report the actual
numbers.)

## Self-review before returning

- Confirm the new check cannot be bypassed: trace that every call site
  reaching `validate_and_render` passes claims that went through either
  `_structured_claims` (now emits `quote`) or `_extractive_claim` (now sets
  `quote`) — no third path constructs a claim dict without one.
  `retrieve_precedent()`/Phase D remains unwired into `/ask` — do not wire
  it in while touching this file.
- Confirm the quote itself never leaks into what gets rendered as a
  citation — `_citation()` still only reads canonical `component`/
  `source_publication` metadata, never the claim or quote text.
- Confirm normalization is symmetric (both `quote` and `chunk_text` go
  through the identical normalization function before comparison) — a
  common bug class here is normalizing one side and not the other.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
