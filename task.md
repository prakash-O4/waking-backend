# task.md — AGENT-29: composer output re-validation

## Objective

`answer_composer_node` (`query_graph.py:381`) sends the whole set of
server-validated claims (`all_results`, straight out of
`validate_and_render`) to a **second, independent** Gemini call
(`_compose_answer`, `gated_orchestrator.py:179`), which reformats them into
the actual final answer JSON — and whatever that call returns becomes
`_response` **directly**, with zero server-side check. Confirmed live gap
(finding #5 in the retrieval-quality review, see `.agent/PROGRESS.md`'s
"Retrieval-quality review + 6-task program" entry): the composer's prompt
schema literally asks the model to write its own `"citation": {}` object
per `relevant_sections` entry, restrained only by the soft instruction
"Never invent law. Never modify citations. Use only what the claims
provide." — advisory text in a prompt, not an enforced check. This is a
direct live risk to Core Invariant #3 ("the model never writes citations")
at the exact point the answer reaches the user — `build_graph()`'s
`answer_composer` node output (`query_graph.py:469`,
`return cast(dict[str, Any], result["_response"])`) **is** the API
response.

Fix: stop trusting the composer's own `"citation"` field entirely. Have it
identify *which validated claim* each section is based on (an id it already
sees verbatim in its own input), and have the **server** — not the model —
resolve and attach the actual citation from `all_results`. Any section that
can't be tied back to a real, non-abstained validated claim is dropped, not
rendered. This is the same "server resolves, model only points" pattern
`validate_and_render` already uses for individual claims — applied one
layer up, at composition.

## Design decisions made here (not open for reinterpretation by the engineer)

1. **Join key is `(evidence_id, as_of)`, not `evidence_id` alone.** A
   diachronic query can carry multiple issues with different per-claim
   `as_of` values (Core Invariant #6 — every claim validates against its
   *declared* as-of, not one session-wide as-of). The same `evidence_id`
   could legitimately appear under two different `as_of` values with
   different citation content (e.g. an amendment applies under one as-of
   and not the other). Keying on `evidence_id` alone would let a section
   silently pick up the wrong `as_of`'s citation. Both fields are already
   present on every entry in `all_results` (`evidence_id`, and `as_of` —
   added by `validate_node`, `query_graph.py:286`) — this costs nothing
   extra to look up.
2. **`abstained` is always server-recomputed, never trusted from the
   composer's own JSON** — same principle as Core Invariant #7 applied one
   layer up. After filtering `relevant_sections` down to only the sections
   that matched a real validated claim, set
   `composed["abstained"] = not composed["relevant_sections"]`
   unconditionally — ignore whatever boolean the model put in its own
   `"abstained"` key.
3. **When the final (post-filter) `relevant_sections` is empty, also blank
   `plain_language`** (`composed["plain_language"] = ""`). Presenting
   prose next to an empty, abstained result is its own quiet-wrong-answer
   risk (`AGENTS.md`'s prime directive) — the text could describe exactly
   the content that just got stripped for lacking real evidence.
4. **When only *some* sections are stripped** (partial match), leave
   `plain_language` as-is. Surgically editing free-form prose to remove
   references to one stripped section is a real, harder problem (would
   need another model call or text-span tracking) — explicitly out of
   scope for this task. Note this as a known, accepted limitation in your
   completion report, don't attempt to solve it.
5. **Scope is citations only** — `conflicts`, `missing_facts`,
   `plain_language`, and `disclaimer` are narrative/descriptive fields, not
   evidence-bearing. Do not add re-validation for them; that's a different,
   much fuzzier problem this task does not cover.

## Acceptance criteria

1. **Prompt change** (`gated_orchestrator.py::_compose_answer`): replace
   the `"citation": {}` field in the `relevant_sections` item schema with
   `"evidence_id": "<the evidence_id from the validated claim this section
   is based on>"` and `"as_of": "<that claim's as_of>"`. Instruct the model
   explicitly to copy these two values verbatim from the claim it's
   summarizing in `VALIDATED CLAIMS` — never invent an id, never modify it.
   Keep the rest of the schema (`section`, `why_applicable`,
   `applicability`, `condition`) unchanged.
2. **New re-validation function** in `gated_orchestrator.py` (near
   `_compose_answer`, e.g. `_revalidate_composed(composed, all_results)`):
   - Build a lookup keyed by `(evidence_id, as_of)` from every entry in
     `all_results` where `abstained` is falsy and both `evidence_id` and
     `citation` are present — value is that entry's `citation` dict
     (canonical, from `validate_and_render`/`_citation()`).
   - For each item in `composed.get("relevant_sections") or []` (defend
     against a missing/malformed key or non-list value — treat as empty):
     skip non-dict items; look up `(item.get("evidence_id"),
     item.get("as_of"))` in the lookup. If found, **replace**
     `item["citation"]` with the canonical citation dict from the lookup
     (never trust anything the model put there, even if it happened to
     match) and keep the item. If not found, drop the item.
   - Recompute `composed["abstained"]` and conditionally blank
     `plain_language` per the design decisions above.
   - Return the mutated (or a new) `composed` dict. Pure function of its
     two inputs — no DB/network access, no LLM call.
3. **Wire it in** (`query_graph.py::answer_composer_node`): call the new
   function on `composed` immediately after `_compose_answer` returns and
   before `composed["query_type"] = query_type; return {"_response":
   composed}` — only on the success path (`composed is not None`). The
   `composed is None` fallback branch (already returns raw
   `all_results` with real per-claim citations, no composer involved) is
   untouched — no re-validation gap exists there.
4. **Adjacent one-line fix, same function, same `abstained`-correctness
   theme**: the `composed is None` fallback branch currently sets
   `"abstained": not all_results` (`query_graph.py:375`) — this is `True`
   only when the list is *empty*, not when every entry in it is
   individually `abstained: True`. A query where every claim abstains but
   `all_results` is still a non-empty list of abstained entries currently
   reports `abstained: False` at the top level, which is wrong and
   contradicts the same "abstained must reflect whether any real evidence
   survived" principle this whole task is about. Fix it to derive from the
   actual per-entry `abstained` flags (e.g. `not any(not r.get("abstained")
   for r in all_results)`), not from list emptiness.
5. No change to `_compose_answer`'s Gemini call itself, its exception
   handling, `_emit_answer_trace_from_state`, or anything in
   `validate_and_render`/`validate_node` — this task only adds a
   server-side filter/rewrite step on the composer's *output*, after it
   returns.

## Branch

`agent/composer-output-revalidation` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: model never writes citations; abstention
  is server-owned; every claim validates against its declared as-of;
  retrieved text/model output is untrusted. Prime directive: a loud
  refusal beats a quiet wrong answer.
- `system-design.md` Core Invariant #3 (model emits claims + evidence_ids
  only, never citations), #6 (as-of is per-claim), #7 (abstention is
  server-owned, model self-abstention is advisory only), #8 (citations
  rendered from canonical metadata, never copied from model output).
- `system-design.md` §14 **PS-7** (abstention is server-owned) — this
  task's primary target, applied at the composition layer rather than the
  per-claim layer AGENT-27 already covers.

## Allowed scope

- `app/retrieval/gated_orchestrator.py` — **only** `_compose_answer`'s
  prompt/schema and the new `_revalidate_composed` function. Do not touch
  `_structured_claims`, `_extractive_claim`, `_citation`-adjacent code, or
  anything else in this file.
- `app/retrieval/query_graph.py` — **only** `answer_composer_node`. Do not
  touch `validate_node`, `reasoner_node`, or anything else in this file.
- `tests/test_orchestrator.py` — the existing test file for
  `gated_orchestrator.py` (already has `test_compose_answer_success` and
  imports it as `orchestrator`). Add new tests there, do not create a
  parallel file.
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not add re-validation for `conflicts`/`missing_facts`/
  `plain_language`/`disclaimer` — citations only, per the design decisions
  above.
- Do not attempt to surgically edit `plain_language` when only some
  sections are stripped — accepted limitation, not this task's job.
- Do not touch `validate_and_render`, `_citation`, `_terminated_before`,
  `eligible_chunk_ids`, `_structured_claims`, `_extractive_claim`, or
  anything precedent-related.
- Do not add a new dependency, service, or abstraction — this is a pure
  post-processing filter function over data already in memory.
- Do not change what `_compose_answer` returns on failure (`None` +
  caller's existing fallback) — only the success-path schema/content.

## Required checks

- `make test`
- `make lint`
- `make eval-gates` — all three zero-tolerance gates
  (`repealed-as-current`, `not-yet-effective-as-current`,
  `overruled-as-good-law`) must stay at `0` — untouched by this task, must
  show no regression.

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

(Guarded in the sense of "must show no regression" — this task doesn't
touch the predicates behind these gates, but it does touch the last-mile
rendering path, so verify nothing here can leak an uncited/unvalidated
claim into a rendered answer.)

## Self-review before returning

- Confirm `_revalidate_composed` never trusts any citation content the
  model produced — every rendered citation must be the exact dict object
  (or an equal copy) that `_citation()` produced for that `(evidence_id,
  as_of)` pair inside `validate_and_render`, not merged with or
  fallback-filled from anything in the composer's own output.
- Confirm a section citing an **abstained** claim's `evidence_id` (the
  composer can see abstained claims in its input — their `citation` is
  `None` but `evidence_id` is still visible) is correctly dropped, not
  rendered with a `None` citation.
- Confirm a section citing an `evidence_id` that doesn't appear in
  `all_results` at all (fabricated/hallucinated by the composer) is
  dropped.
- Confirm the `(evidence_id, as_of)` compound-key case: seed two entries
  in `all_results` with the same `evidence_id` but different `as_of` and
  different `citation` content; prove a composed section citing one
  `as_of` gets that `as_of`'s citation, not the other's.
- New tests must prove all of the above with real function calls on
  constructed `all_results`/`composed` fixtures, not canned booleans.
- Report which existing tests (if any exercise `_compose_answer`/
  `answer_composer_node` today) you had to update, and why.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
