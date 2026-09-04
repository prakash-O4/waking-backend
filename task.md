# AGENT-36: claim-support gap measurement tooling

## Objective

Roadmap item 4 (see `.agent/PROGRESS.md`) requires measuring the real size
of the "quote exists but claim doesn't follow" gap using labeled eval data,
**before** deciding whether to add an entailment/NLI/LLM-judge gate. Do not
build any entailment model — this task only builds the measurement.

`scripts/label_eval_candidates.py` (AGENT-34) already writes
`app/eval/golden/claim_support.json` rows shaped
`{query, as_of, quote, claim, evidence_id, supports, gate_verdict_abstained,
source, labeled_by, labeled_at}`. That file is currently empty — no labels
exist yet.

**Problem**: `gate_verdict_abstained` is not a clean proxy for "did the
quote-support check pass." `validate_and_render()`
(`app/retrieval/validation_gate.py:163-187`) sets `abstained=True` if *any*
of five independent checks fails: hash integrity, eligibility, termination
(`_terminated_before`), expression staleness (`_expression_stale`), **or**
the quote-substring check (`_claim_supported`, line 23-25). A claim can be
`abstained=True` purely because the cited chunk is stale or ineligible —
nothing to do with whether the quote supports the claim. Comparing human
`supports` labels against this blanket boolean would misattribute unrelated
gate failures as claim-support failures.

## Required change 1: isolate the quote-check verdict per claim

In `scripts/label_eval_candidates.py`:

- Import `_expression` and `_claim_supported` directly from
  `app.retrieval.validation_gate`, alongside the existing
  `validate_and_render` import. (Precedent: this file already imports
  `_structured_claims` directly from `gated_orchestrator.py` — same
  pattern, not a new one.)
- `_run_pipeline()` currently opens `conn` in a `with` block, calls
  `retrieve_postgres` → `_structured_claims` → `validate_and_render`, then
  returns `(hits, claims, rendered)` after the connection closes. While
  `conn` is still open, also compute one isolated boolean per claim:
  `_expression(conn, claim["evidence_id"], as_of)` → if it returns `None`,
  `quote_check_passed = False` (fails closed, matches `validate_and_render`'s
  own default when `expr` is missing); otherwise
  `quote_check_passed = _claim_supported(claim["quote"], expr[0])`.
  Return this as a 4th list, same order/length as `claims`:
  `(hits, claims, rendered, quote_check_passed)`.
- Update `show()` to unpack 4 values (the 4th can be unused there, or
  printed alongside each claim line — your call, not required).
- Update `label()` to unpack 4 values. When writing each `claim_support.json`
  row, add a new field: `"quote_check_passed": quote_check_passed[idx]`.

## Required change 2: `--report` mode

Add `--report` as a new mutually-exclusive top-level mode (alongside
`--fetch/--list/--show/--label/--skip`) that takes no other required args.
It must:

- Read `claim_support.json` only (no DB connection, no Langfuse client).
- If empty/missing, print a clear "no labeled claims yet" message and
  return — do not crash or divide by zero.
- Otherwise print a 2x2 breakdown over `supports` (human label) ×
  `quote_check_passed` (isolated automated verdict):
  - `supports=True, quote_check_passed=True` — count
  - `supports=True, quote_check_passed=False` — count (quote-matching false
    negative — e.g. whitespace/OCR drift — not the gap this roadmap item
    targets, but worth surfacing)
  - `supports=False, quote_check_passed=True` — count. **This is the number
    roadmap item 4 exists to measure**: quote passes verbatim-match but the
    claim doesn't actually follow from it.
  - `supports=False, quote_check_passed=False` — count
  - Total labeled claims.
  - The gap rate: `(supports=False & quote_check_passed=True) /
    (quote_check_passed=True total)`, as a percentage — print `"n/a (no
    quote_check_passed=True rows)"` instead of dividing by zero if that
    denominator is 0.
- Rows written before this task (i.e. missing `quote_check_passed` — none
  exist today, `claim_support.json` is currently empty, so no migration
  path is needed; do not build one) — if you want defensive handling for
  a hypothetical old row missing the key, treat it as
  `quote_check_passed=False`, but this is not a required test case since
  no such data exists.

## Explicitly forbidden

- Touching `validation_gate.py`, `eligibility_gate.py`,
  `gated_orchestrator.py`, `postgres_retriever.py`, or `main.py` — import
  and call `_expression`/`_claim_supported` as-is, do not modify their
  logic or signatures.
- Touching `_MIN_QUOTE_CHARS` or `_normalize` — that tunable is explicitly
  flagged in code as "revisit with eval data"; this task produces the eval
  data, but changing the threshold itself is a follow-on decision after
  Prakash reviews `--report` output, not part of this task.
- Adding any entailment/NLI/LLM-judge model, or any new automated
  claim-support check — out of scope until the `--report` measurement
  shows a real, sized problem (Prakash's call, not this task's).
- Any auto-labeling or heuristic guessing of `supports` — labels stay
  100% human via `--label`/`--by`, unchanged.
- New dependencies, new files, new DB tables/columns. Everything here is
  an edit to the existing `scripts/label_eval_candidates.py`.
- Writing any sample/fake data into `claim_support.json` or
  `labeled_traffic.json` — those are Prakash's own labeling outputs, not
  yours to populate.

## Required tests (add to `tests/test_label_eval_candidates.py`, follow its
existing `patch_pipeline`/`seed_queue`/`files` fixture conventions)

1. `label()` with a claim whose quote is present verbatim (≥15 chars) in
   the mocked expression text records `"quote_check_passed": true`.
2. `label()` with a claim whose quote does NOT appear in the mocked
   expression text (or is too short) records `"quote_check_passed": false`.
3. `label()` when `_expression()` returns `None` for the claim's
   `evidence_id` records `"quote_check_passed": false` and does not raise.
4. `--report` (call the underlying function directly, e.g. `lec.report()`)
   with an empty/missing `claim_support.json` prints a no-data message and
   does not raise.
5. `--report` with a seeded mix of all 4 quadrants prints the correct
   counts for each quadrant and the correct gap-rate percentage (hand-
   compute the expected numbers in the test).
6. `--report` with labeled rows where `quote_check_passed=True` total is 0
   prints the "n/a" denominator case instead of raising `ZeroDivisionError`.

## Required checks

- `make test`
- `scripts/label_eval_candidates.py` is not in the Makefile's fixed lint
  file list (pre-existing gap, same as every prior new/modified script
  this session — do not fix the Makefile, just also run manually):
  `python3 -m ruff check scripts/label_eval_candidates.py && python3 -m ruff format --check scripts/label_eval_candidates.py && python3 -m mypy --strict --follow-imports=skip --disable-error-code=misc --disable-error-code=import-untyped scripts/label_eval_candidates.py`
- `make lint`

No `make eval-gates` relevance — this task touches no gate/temporal/
ingestion/precedent path; it's an offline, human-invoked script never
imported by any serve-path module (same conclusion AGENT-34's task.md
reached for this same file).

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line. Use:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.

## Branch

`agent/claim-support-report`, off `dev`.

## After merge (not part of this task)

Building this tool does not by itself answer roadmap item 4. Prakash must
run `--fetch` against real Langfuse traffic and personally `--label` a
meaningful number of real (quote, claim) pairs over time before
`--report`'s numbers mean anything. The claim-support verifier decision
waits on that real data, not on this tooling PR alone.
