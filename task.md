# task.md — AGENT-42: fix reasoner JSON-parse fallback silently producing wrong-context answers

## How this was found

Prakash posed a real question through the console ("What rights a
labor/employee have in the workspace of their?") and pasted the
Langfuse trace. The `structured_claims` generation span shows the
reasoner (`_structured_claims` in `app/retrieval/gated_orchestrator.py`)
actually found the right evidence — a claim citing दफा ५१ of श्रम ऐन
२०७४ ("leave beyond sick/bereavement/maternity is a privilege, not a
right", evidence_id `c9335ea1-...`), correctly on-topic for the
question. But the raw model output has a malformed `quote` field:

```
"quote": \"(१) यस परिच्छेद बमोजिम...
```

— a stray literal backslash before the opening quote, which is invalid
JSON. `json.loads` throws `Expecting value: line 1 column 257 (char
256)`, caught by `_structured_claims`'s bare `except Exception`, logged
via `_log_generation_fallback`, and the function returns `None`.

The caller (`reasoner_node` in `app/retrieval/query_graph.py:207-210`)
treats `None` as "no result" and falls back to
`_orch._extractive_claim(issue_hits)`
(`app/retrieval/gated_orchestrator.py:129-139`), which unconditionally
takes `hits[0]` — whatever ranked first in retrieval — truncates it to
300 chars, and presents it as both `claim` and `quote`, with **no
relevance check at all**. In this trace `hits[0]` was दफा ९ ("who
decides if someone is unemployed") — completely off-topic for a
"what rights does a worker have" question. The system answered
confidently (`abstained: false`) with the wrong section instead of
either recovering the correct claim or abstaining.

This is a direct violation of `AGENTS.md`'s prime directive ("a loud
refusal beats a quiet wrong answer") and Core Invariant 7 in
`system-design.md` §2 ("Abstention is server-owned... the model's
self-abstention is an advisory prior only"): a recoverable JSON
formatting glitch silently became a wrong, non-abstained answer instead
of the honest failure mode (retry, or abstain).

## Root cause (confirmed empirically against the live Azure resource)

Azure OpenAI chat completions, called with no `response_format`
constraint, can emit syntactically invalid JSON (this exact
stray-backslash-before-quote pattern) when asked to echo a verbatim
substring in a JSON string field. This is a known, well-documented
failure mode of unconstrained JSON-via-prompt, and Azure/OpenAI's
`response_format={"type": "json_object"}` (JSON mode) exists
specifically to eliminate it at the API level — the API guarantees
syntactically valid JSON output when this is set.

Verified directly against the real `gpt-4.1-mini` deployment (not just
read from docs): a script constructing `AzureChatOpenAI(...,
model_kwargs={"response_format": {"type": "json_object"}})` and asking
the model to echo a value containing an embedded double-quote produced
correctly-escaped, valid JSON — parsed cleanly with `json.loads`. This
worked at API versions `"2025-01-01-preview"`, `"2024-10-21"` (GA), and
`"2024-08-01-preview"` — all confirmed live. **`"2024-10-21"` (GA, not
preview) is the version to use** — no reason to pin to a preview
version when a GA one is confirmed working.

**Important scoping finding**: `AZURE_OPENAI_API_VERSION`
(`app/config.py:39`, default `"2023-05-15"`) is a **shared** setting —
also used by `app/ingestion/metadata_enricher.py`,
`app/ingestion/pgvector_indexer.py`, `app/eval/ragas_eval.py`, and
`app/retrieval/postgres_retriever.py`'s `translate_query()` and
`_embed_query()`. Bumping it globally would touch ingestion, eval, and
embeddings paths that have nothing to do with this bug and were not
verified against the new API version. **Do not touch it.** Instead, add
a **new, dedicated** setting scoped only to this fix.

## Objective

Stop this bug class at the root — make the Azure OpenAI JSON calls in
the reasoner/composer path request `response_format={"type":
"json_object"}` — while keeping the existing fallback behavior intact
as a safety net for any *other* kind of failure (network error, empty
response, schema mismatch).

## Acceptance criteria

1. In `app/config.py`, add a new setting:
   `AZURE_OPENAI_LLM_API_VERSION: str = "2024-10-21"` — placed next to
   the existing `AZURE_OPENAI_LLM_*` block (`app/config.py:42-45`). Do
   **not** change `AZURE_OPENAI_API_VERSION` (line 39) or any of its
   other call sites.

2. In `app/retrieval/gated_orchestrator.py`, all three
   `AzureChatOpenAI(...)` constructions — `_structured_claims`
   (~line 197), `_compose_answer` (~line 284), `_fact_extract`
   (~line 372) — must:
   - use `api_version=s.AZURE_OPENAI_LLM_API_VERSION` instead of
     `s.AZURE_OPENAI_API_VERSION`;
   - pass `model_kwargs={"response_format": {"type": "json_object"}}`.

3. `_structured_claims` currently parses the response with
   `cast(dict[str, Any], json.loads(str(resp.content).strip()))` —
   inconsistent with its two siblings, which use
   `llm_text(resp).strip()` then `json.loads(_json_payload(raw))`
   (`llm_text` in `app/utils/llm.py`, `_json_payload` in
   `gated_orchestrator.py:89-95`, strips ` ```json ` fences). Bring
   `_structured_claims` in line with the same two-step pattern its
   siblings already use, so all three behave identically. This is a
   pure consistency fix (existing helpers, no new code) — no other
   change to `_structured_claims`'s control flow, signature, or return
   shape.

4. Do **not** touch `_extractive_claim()`
   (`gated_orchestrator.py:129-139`) or its fallback behavior. Whether
   the extractive fallback itself should have a relevance check, or
   abstain instead of guessing, is a separate, bigger design question —
   explicitly out of scope here. This task only makes the *primary*
   path (the LLM call) more reliable so the fallback triggers less
   often; it does not change what the fallback does when it does
   trigger.

5. Tests (`tests/test_orchestrator.py`): add or extend coverage
   asserting `response_format={"type": "json_object"}` and
   `api_version=<the new setting>` are actually passed to
   `AzureChatOpenAI` for all three functions (mock
   `AzureChatOpenAI` and inspect the call kwargs, matching this file's
   existing mocking patterns — see `test_structured_claims_success` at
   line 561 for the existing shape of this test). Every existing test
   that exercises the `_structured_claims is None` → extractive
   fallback path, the `_compose_answer is None` → raw-claims fallback
   path, and the `_fact_extract` exception → passthrough fallback path
   must keep passing unmodified in behavior — this task adds a
   reliability improvement upstream, it does not remove the safety net
   downstream.

## Explicitly forbidden

- Touching `AZURE_OPENAI_API_VERSION` (the shared setting) or any of
  its other call sites: `app/ingestion/metadata_enricher.py`,
  `app/ingestion/pgvector_indexer.py`, `app/eval/ragas_eval.py`,
  `app/retrieval/postgres_retriever.py` (`translate_query`,
  `_embed_query` — neither emits JSON, neither is affected by this bug
  class, both stay exactly as they are).
- Touching `_extractive_claim()`, `query_graph.py`,
  `validation_gate.py`, `eligibility_gate.py`, or any gate/temporal
  logic.
- Adding any retry loop, backoff, or new dependency — `response_format`
  alone is the fix; don't add speculative complexity on top of it.
- Changing `_structured_claims`/`_compose_answer`/`_fact_extract`'s
  external return shape, or the JSON schema described in their system
  prompts.

## Governing references

- `AGENTS.md` prime directive: "A loud refusal beats a quiet wrong
  answer."
- `system-design.md` §2, Core Invariant 7: "Abstention is server-owned."
- PS-7 (abstention is server-owned; model self-abstention is advisory
  only) — this task does not touch gate logic, but the bug it fixes is
  a real-world instance of that invariant being violated in practice by
  an untrusted-generation formatting failure, not a gate defect.

## Required checks

- `make test`
- `make lint`
- Manual smoke test against the **real** Azure endpoint (same
  empirical-verification standard as AGENT-41/AGENT-40 — this codebase
  has been burned before by trusting an LLM API's documented shape
  without checking it live): construct the actual
  `AzureChatOpenAI(..., model_kwargs={"response_format": {"type":
  "json_object"}})` call as it will really run in
  `_structured_claims`, invoke it with a real prompt, confirm valid
  JSON comes back. (Claude already verified the bare mechanism works
  against this exact deployment/API version standalone — this step is
  to confirm the *wired-in* code path, not to re-derive the API's
  behavior from scratch.)

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, Pi, or any AI
attribution. Enforce via:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`
No `Co-Authored-By` trailers, no "Generated with" lines.
