# task.md — AGENT-46: scale retrieval breadth for enumerate/list-style questions

## How this was found

Prakash asked "List me the basic rights of labor" and got back only 2
short citations (दफा ३, दफा ९) even though श्रम ऐन २०७४ (the Labour Act)
alone has 150+ substantive sections covering wages, hours, leave,
safety, termination, and dispute resolution. Investigation (grounded in
the actual code, not guessed) confirmed a real, structural gap: `k=5`
(top-5 retrieval) is a hardcoded constant applied identically to every
query regardless of phrasing (`app/retrieval/query_graph.py:126-128,
149-151`), and `_fact_extract`'s issue-query decomposition caps at
`MAX_SUBQUERIES = 3` (`app/retrieval/gated_orchestrator.py:19`) with no
mechanism anywhere that detects "list all X" / enumerate-style phrasing
and scales breadth in response. A narrow lookup ("what is the notice
period for termination") and a broad enumerate question ("list all
rights") are currently retrieved by the exact same mechanism.

Prakash separately raised a bigger future direction — a single
orchestrating "brain" agent that decides *how* to handle each query,
routing to the right retrieval/reasoning strategy — but explicitly said
"maybe not now, we need to evaluate the usecase first." This task is
**not** that. It's a deliberately narrow, interim fix: teach the
existing `_fact_extract` LLM call (which already decomposes questions
into issue_queries) to decompose *more* when the question is genuinely
broad, reusing infrastructure that already exists rather than building
a new routing layer.

## Design, verified empirically before writing this brief

**Verified against the real Azure `gpt-4.1-mini` deployment** (not
assumed): a soft instruction ("split into multiple issue_queries for
broad questions") was tried first and the model ignored it — one
issue_query came back for "List me the basic rights of labor" anyway.
A stronger, more directive version was then tested and worked
correctly on both a broad and a narrow control question:

- "List me the basic rights of labor" → 6 issue_queries, one each for
  पारिश्रमिक (wages), काम गर्ने समय (working hours), बिदा (leave), कार्यस्थल
  सुरक्षा (workplace safety), सेवा अन्त्य (termination), विवाद समाधान
  (dispute resolution).
- "What are all the obligations of an employer under the Labour Act?"
  → 6 issue_queries, same sub-topic split, correctly reframed for
  employer obligations instead of employee rights.
- "What is the notice period for termination under the Labour Act"
  (narrow control) → exactly 1 issue_query, unchanged.

**Real cost this introduces, addressed directly in this task, not
ignored**: `reasoner_node` (`app/retrieval/query_graph.py:179-218`)
currently calls `_orch._structured_claims()` **sequentially**, once per
issue_query, in a plain `for` loop. Each call already takes 3-14
seconds against the real API (see AGENT-42/43/44 verification notes in
this file). Going from 1 to up to 6 issue_queries would multiply
reasoner latency by up to 6x if left sequential — unacceptable.
`retrieve_node` (`app/retrieval/query_graph.py:114-176`) already has a
proven `ThreadPoolExecutor`-based parallel fan-out for exactly this
situation (multiple issue_queries), reusing the already-imported
`ThreadPoolExecutor` and the already-defined `MAX_RETRIEVER_FANOUT = 5`
constant (`query_graph.py:4,18`), with a documented convention: **don't
pass `lf_trace` into the parallel workers** ("Langfuse trace objects
are not assumed thread-safe; node-level traces stay on main thread",
`query_graph.py:150`) and fall back to the sequential path on any
exception. This task extends the exact same pattern to `reasoner_node`
— no new concurrency primitive, no new dependency, reusing what already
works.

## Objective

1. Raise `MAX_SUBQUERIES` and strengthen `_fact_extract`'s
   decomposition instruction so broad/enumerate questions get split
   into multiple topic-specific issue_queries, while narrow questions
   are unaffected.
2. Parallelize `reasoner_node` across issue_queries (mirroring
   `retrieve_node`'s existing pattern exactly) so the added breadth
   doesn't multiply wall-clock latency.

## Part A — `app/retrieval/gated_orchestrator.py`

1. Change `MAX_SUBQUERIES = 3` (line 19) to `MAX_SUBQUERIES = 6`.

2. Replace the `_fact_extract` system prompt's decomposition
   instruction (currently `gated_orchestrator.py:370-373`:
   ```python
       f"Default as_of when not specified: {session_as_of.isoformat()}. "
       f"Max {MAX_SUBQUERIES} issue_queries. "
       "Write issue_queries in formal Devanagari Nepali for best embedding match."
   ```
   with the empirically-verified version:
   ```python
       f"Default as_of when not specified: {session_as_of.isoformat()}. Max {MAX_SUBQUERIES} issue_queries.\n"
       "IMPORTANT — broad/enumerate questions (asking to 'list', 'what are all', "
       "'सूची', 'सबै', or otherwise survey an entire topic rather than one specific "
       "fact) MUST be split into 4-6 issue_queries, one per distinct legal sub-topic "
       "(e.g. for labor rights: wages/पारिश्रमिक, working hours/काम गर्ने समय, leave/बिदा, "
       "workplace safety/सुरक्षा, termination/सेवा अन्त्य, dispute resolution/विवाद समाधान). "
       "A single issue_query is WRONG for this question type — you must enumerate the "
       "sub-topics yourself and issue one query per sub-topic. "
       "A narrow, specific-fact question still gets exactly one issue_query as before.\n"
       "Write issue_queries in formal Devanagari Nepali for best embedding match."
   ```
   (The sub-topic examples are illustrative for the model, not a fixed
   taxonomy — keep them as-is, they were part of what was verified
   working.)

Do **not** change `k` (stays `5` everywhere) — this task scales the
*number* of targeted sub-queries, not the chunk count per sub-query.
Do **not** change `_compose_answer`'s prompt or its `max_tokens=2048` —
out of scope; if a genuinely large `relevant_sections` list ever proves
to need more room, that's a separate, measured follow-up, not bundled
here.

## Part B — `app/retrieval/query_graph.py`: parallelize `reasoner_node`

Current (`query_graph.py:179-218`):
```python
def reasoner_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    """Structured reasoning over authority-ranked context, one LLM call per issue."""
    lf_trace = config["configurable"].get("lf_trace")
    all_hits = state["all_hits"]
    if not all_hits:
        return {}

    issue_queries: list[dict[str, Any]] = state["issue_queries"] or [
        {
            "query": state["raw_query"],
            "as_of": state["session_as_of"],
            "work_type_hint": None,
        }
    ]
    facts = state["facts"]
    query_type = state["query_type"]
    pending_results: list[dict[str, Any]] = []

    hits_by_issue: dict[int, list[dict[str, Any]]] = {}
    for h in all_hits:
        idx = h.get("_issue_idx", 0)
        hits_by_issue.setdefault(idx, []).append(h)

    for idx, iq in enumerate(issue_queries):
        issue_hits = hits_by_issue.get(idx, [])
        if not issue_hits:
            continue

        parsed = _orch._structured_claims(facts, [iq], issue_hits, lf_trace=lf_trace)
        if parsed is None:
            claims = _orch._extractive_claim(issue_hits)
            query_type = "extractive"
        elif parsed.get("abstain") or not parsed.get("claims"):
            continue
        else:
            claims = parsed["claims"]

        pending_results.append({"claims": claims, "as_of": iq["as_of"]})

    return {"query_type": query_type, "_pending_results": pending_results}
```

Replace with (mirrors `retrieve_node`'s exact structure — sequential
helper reused both as the `len(issue_queries) <= 1` fast path and as
the exception fallback, exactly like `retrieve_node` does):
```python
def reasoner_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    """Structured reasoning over authority-ranked context, one LLM call per issue."""
    lf_trace = config["configurable"].get("lf_trace")
    all_hits = state["all_hits"]
    if not all_hits:
        return {}

    issue_queries: list[dict[str, Any]] = state["issue_queries"] or [
        {
            "query": state["raw_query"],
            "as_of": state["session_as_of"],
            "work_type_hint": None,
        }
    ]
    facts = state["facts"]
    base_query_type = state["query_type"]

    hits_by_issue: dict[int, list[dict[str, Any]]] = {}
    for h in all_hits:
        idx = h.get("_issue_idx", 0)
        hits_by_issue.setdefault(idx, []).append(h)

    def run_one(
        idx: int, iq: dict[str, Any], trace: Any
    ) -> tuple[int, dict[str, Any] | None]:
        issue_hits = hits_by_issue.get(idx, [])
        if not issue_hits:
            return idx, None

        parsed = _orch._structured_claims(facts, [iq], issue_hits, lf_trace=trace)
        if parsed is None:
            claims = _orch._extractive_claim(issue_hits)
            return idx, {"claims": claims, "as_of": iq["as_of"], "extractive": True}
        if parsed.get("abstain") or not parsed.get("claims"):
            return idx, None
        return idx, {
            "claims": parsed["claims"],
            "as_of": iq["as_of"],
            "extractive": False,
        }

    def sequential() -> list[tuple[int, dict[str, Any] | None]]:
        return [run_one(idx, iq, lf_trace) for idx, iq in enumerate(issue_queries)]

    if len(issue_queries) <= 1:
        results = sequential()
    else:
        try:
            max_workers = min(len(issue_queries), MAX_RETRIEVER_FANOUT)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Langfuse trace objects are not assumed thread-safe; node-level traces stay on main thread.
                futures = [
                    executor.submit(run_one, idx, iq, None)
                    for idx, iq in enumerate(issue_queries)
                ]
                results = [f.result() for f in futures]
        except Exception as e:
            _trace_error(lf_trace, "parallel_reasoning", e)
            results = sequential()

    pending_results: list[dict[str, Any]] = []
    used_extractive = False
    for idx, res in sorted(results, key=lambda r: r[0]):
        if res is None:
            continue
        used_extractive = used_extractive or res.pop("extractive")
        pending_results.append(res)

    query_type = "extractive" if used_extractive else base_query_type
    return {"query_type": query_type, "_pending_results": pending_results}
```

`ThreadPoolExecutor` and `MAX_RETRIEVER_FANOUT` are already imported/
defined at the top of this file — no new import needed.

## Acceptance criteria

- `MAX_SUBQUERIES = 6`, prompt updated exactly as specified in Part A.
- `reasoner_node` parallelized exactly as specified in Part B —
  `query_type` becomes `"extractive"` if **any** issue used the
  extractive fallback (matches the original sequential code's behavior
  exactly: once set, it was never reset back), `pending_results`
  preserves issue-index order regardless of thread completion order,
  parallel workers never receive `lf_trace` (pass `None`), any
  exception during parallel dispatch falls back to the sequential path
  with `_trace_error(lf_trace, "parallel_reasoning", e)` logged first.
- New tests in `tests/test_orchestrator.py` (the correct file — no
  dedicated `test_query_graph.py` exists; `test_orchestrator.py:1139`,
  `test_parallel_retrieve_preserves_issue_order_and_closes_pool`, is
  the existing test for `retrieve_node`'s equivalent parallel path —
  mirror its structure/mocking approach directly for `reasoner_node`):
  - `len(issue_queries) <= 1` still takes the sequential path (no
    `ThreadPoolExecutor` involved) — same as before.
  - Multiple issue_queries with all succeeding produces
    `pending_results` in issue-index order, `query_type` unchanged from
    input.
  - One issue's `_structured_claims` returning `None` (mocked) among
    several succeeding issues still sets `query_type == "extractive"`
    and still includes results from the other, successful issues.
  - A mocked exception raised inside the parallel dispatch path falls
    back to the sequential path and still returns correct results.
  - Mocked `_structured_claims` call in the multi-issue-query case
    receives `lf_trace=None`, not the real trace object (assert on the
    mock's call kwargs).
- Existing tests for `_fact_extract` and `reasoner_node`/`stream_query`
  must keep passing — update any test that hardcodes the old
  `MAX_SUBQUERIES = 3` or the old prompt text if one exists.
- `make test`/`make lint` — use `.venv/bin/python`, not bare `python3`.
- Manual live verification (same standard as every prior task this
  session): run "List me the basic rights of labor" against the real
  server/console, confirm multiple issue_queries fire, confirm total
  wall-clock latency for the reasoner stage stays close to a
  single-issue latency (not ~6x), confirm the final answer's
  `relevant_sections` now covers meaningfully more of the Act than the
  2 sections it returned before.

## Explicitly forbidden

- Do not change `k` (stays 5) anywhere.
- Do not touch `retrieve_node`'s own parallel-fan-out implementation —
  reuse its pattern, don't modify it.
- Do not touch `_compose_answer`, its prompt, or `max_tokens`.
- Do not add a new concurrency primitive, thread pool class, or
  dependency — `ThreadPoolExecutor` and `MAX_RETRIEVER_FANOUT` already
  exist in this file; reuse them exactly.
- Do not build any form of query-routing/"orchestrator agent" — that
  bigger direction is explicitly deferred per Prakash, not this task.
- Do not touch `validate_node`, `validation_gate.py`, or
  `eligibility_gate.py` — this task only changes how many issue_queries
  get generated and how they're processed in parallel, not any gate
  logic.

## Governing references

No Core Invariant or PS-* requirement is weakened — the model still
only emits `claims + evidence_ids` (Core Invariant 3), the validation
gate still runs identically per claim regardless of how many claims
there are (Core Invariant 4/7). This task changes retrieval/reasoning
*breadth*, not the gate that verifies claims afterward.

## Required checks

- `.venv/bin/python -m pytest tests/`
- `.venv/bin/python -m ruff check` / `ruff format --check` / `mypy
  --strict` on the exact Makefile file list (substitute
  `.venv/bin/python` for `python3`)
- Manual live verification as described above

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, Pi, or any AI
attribution. Enforce via:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`
No `Co-Authored-By` trailers, no "Generated with" lines.
