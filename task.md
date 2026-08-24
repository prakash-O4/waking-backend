# Task: AGENT-2 — Fact Extractor + issue-driven retrieval (Stage 2)

**Branch:** `agent/stage-2-fact-extractor`
**Base:** `dev`
**Engineer:** Pi
**ADR:** `docs/adr-001-multi-agent-query-architecture.md` §Node 1 — Fact Extractor, §Node 2 — Parallel Retriever

---

## Objective

Activate the first two real agents in the LangGraph graph:

1. **Fact Extractor node** — replaces `classify_node`. Calls Gemini 2.5 Flash to extract structured facts, classify missing facts (`required | clarifying | informational`), and produce one targeted retrieval query per legal issue (max 5).
2. **Issue-driven retrieval** — replaces the single-subquery loop in `retrieve_generate_node`. The node now iterates over `issue_queries` (from Fact Extractor) instead of `subqueries`.

**Also included:** Remove `_graph_clock` from `query_graph.py` (cleanup flagged in PROGRESS.md — CPython-specific frame-walking, now unnecessary).

Behaviour is meaningfully different but all existing correctness invariants are unchanged. Eligibility gate, dual-path translation, RRF fusion, reranker, validation gate — all run identically inside each issue branch.

Stage 3 (Authority Ranker + Cross-Ref Resolver) is a separate task. Do not start it here.

---

## Acceptance criteria

1. `orchestrator.answer(question, session_as_of, conn)` signature and return format are unchanged.
2. All 6 tests in `tests/test_orchestrator.py` pass (tests 1–3 updated; tests 4–6 unchanged or updated for new mock target; all passing).
3. Fact extractor gracefully degrades: any exception or missing GEMINI_API_KEY → returns `{facts: None, missing_facts: [], issue_queries: [{query: raw_query, as_of: session_as_of, work_type_hint: None}]}`. Pipeline continues as current baseline.
4. `_graph_clock` removed from `query_graph.py`. `run_query` uses `_orch.time.monotonic()` for `wall_clock_start` (so test mock still works).
5. Two new tests added covering: `_fact_extract` success path (mock Gemini), `_fact_extract` failure fallback.
6. `make test` green (55 → 57 passing), `make lint` clean.
7. Zero-tolerance gates unaffected: repealed-as-current = 0, not-yet-effective-as-current = 0.

---

## Exact scope

**Modified files:**
- `app/retrieval/query_graph.py` — replace `classify_node` with `fact_extractor_node`; update `retrieve_generate_node` to iterate `issue_queries`; remove `_graph_clock`; update `run_query`
- `app/retrieval/gated_orchestrator.py` — add `_fact_extract()` function
- `tests/test_orchestrator.py` — update tests 1–3 to mock `_fact_extract`; add 2 new tests

**No other files modified.** Do NOT touch `query_state.py`, `postgres_retriever.py`, `eligibility_gate.py`, `validation_gate.py`, `reranker.py`, or any eval file.

---

## Implementation guide

### 1. `app/retrieval/gated_orchestrator.py` — add `_fact_extract`

Add this function after `_classify_and_decompose`. Do NOT remove `_classify_and_decompose` (it is still tested directly by `test_classifier_failure_falls_back_to_simple`).

```python
def _fact_extract(question: str, session_as_of: date) -> dict[str, Any]:
    """Extract structured facts and per-issue retrieval queries using Gemini 2.5 Flash.

    Failure mode: any exception or missing key → single raw query passthrough.
    """
    _FALLBACK: dict[str, Any] = {
        "facts": None,
        "missing_facts": [],
        "issue_queries": [{"query": question, "as_of": session_as_of, "work_type_hint": None}],
    }
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return _FALLBACK
    system = (
        "You are a Nepali legal assistant. Analyse the user's legal query and output JSON only:\n"
        '{\n'
        '  "facts": {"parties": [], "events": [], "dates": [], "location": null},\n'
        '  "missing_facts": [\n'
        '    {"fact": "<what is missing>", "type": "required|clarifying|informational"}\n'
        '  ],\n'
        '  "issue_queries": [\n'
        '    {"query": "<Nepali retrieval query>", "as_of": "<YYYY-MM-DD or null>",\n'
        '     "work_type_hint": "<Act|Rule|Regulation|null>"}\n'
        '  ]\n'
        '}\n'
        f"Default as_of when not specified: {session_as_of.isoformat()}. "
        f"Max {MAX_SUBQUERIES} issue_queries. "
        "Write issue_queries in formal Devanagari Nepali for best embedding match."
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=s.GEMINI_API_KEY,
            temperature=0.0,
            max_output_tokens=1000,
        )
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ]
        )
        parsed = cast(dict[str, Any], json.loads(str(resp.content).strip()))

        issue_queries: list[dict[str, Any]] = []
        for iq in parsed.get("issue_queries", [])[:MAX_SUBQUERIES]:
            try:
                as_of = (
                    date.fromisoformat(iq["as_of"]) if iq.get("as_of") else session_as_of
                )
            except (ValueError, TypeError):
                as_of = session_as_of
            issue_queries.append(
                {
                    "query": iq.get("query", question),
                    "as_of": as_of,
                    "work_type_hint": iq.get("work_type_hint"),
                }
            )

        return {
            "facts": parsed.get("facts"),
            "missing_facts": parsed.get("missing_facts", []),
            "issue_queries": issue_queries or _FALLBACK["issue_queries"],
        }
    except Exception:
        return _FALLBACK
```

**Important:** `_FALLBACK` is a `dict`, not a module-level constant — it references `question` and `session_as_of` which are local. Write it as a local variable inside the function as shown above (not as a module-level constant).

---

### 2. `app/retrieval/query_graph.py` — full rewrite

Replace the entire file with this:

```python
from __future__ import annotations

import time
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import app.retrieval.gated_orchestrator as _orch
from app.retrieval.query_state import QueryState


# ── nodes ──────────────────────────────────────────────────────────────────────

def fact_extractor_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    result = _orch._fact_extract(state["raw_query"], state["session_as_of"])
    issue_queries = result["issue_queries"]
    return {
        "facts": result["facts"],
        "missing_facts": result["missing_facts"],
        "issue_queries": issue_queries,
        "query_type": "simple" if len(issue_queries) == 1 else "complex",
    }


def retrieve_generate_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    all_hits: list[dict[str, Any]] = []
    partial_results: list[dict[str, Any]] = []
    query_type = state["query_type"]

    issue_queries = state["issue_queries"] or [
        {"query": state["raw_query"], "as_of": state["session_as_of"], "work_type_hint": None}
    ]

    for iq in issue_queries:
        if _orch._wall_clock_expired(state["wall_clock_start"]):
            break

        query_text: str = iq["query"]
        as_of: date = iq["as_of"]

        hits = _orch.retrieve_postgres(conn, query_text, as_of, k=5)
        all_hits.extend(hits)
        if not hits:
            continue

        parsed = _orch._model_claims(query_text, hits)
        if parsed is None:
            claims = _orch._extractive_claim(hits)
            query_type = "extractive"
        elif parsed.get("abstain") or not parsed.get("claims"):
            continue
        else:
            claims = parsed["claims"]

        partial_results.append({"claims": claims, "as_of": as_of})

    return {
        "all_hits": all_hits,
        "query_type": query_type,
        "_pending_results": partial_results,
    }


def validate_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    all_results: list[dict[str, Any]] = []

    for pending in state.get("_pending_results", []):
        validated = _orch.validate_and_render(pending["claims"], pending["as_of"], conn)
        for result in validated:
            result["as_of"] = pending["as_of"].isoformat()
        all_results.extend(validated)

    return {"all_results": all_results}


def assemble_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    all_results = state["all_results"]
    all_hits = state["all_hits"]
    session_as_of = state["session_as_of"]
    raw_query = state["raw_query"]
    query_type = state["query_type"]

    _orch._emit_answer_trace_from_state(
        raw_query,
        session_as_of,
        query_type,
        all_results,
        all_hits,
        state["wall_clock_start"],
    )

    return {
        "_response": {
            "as_of": session_as_of.isoformat(),
            "query_type": query_type,
            "abstained": not all_results,
            "results": all_results,
        }
    }


# ── graph ──────────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("fact_extractor", fact_extractor_node)
    builder.add_node("retrieve_generate", retrieve_generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("assemble", assemble_node)

    builder.add_edge(START, "fact_extractor")
    builder.add_edge("fact_extractor", "retrieve_generate")
    builder.add_edge("retrieve_generate", "validate")
    builder.add_edge("validate", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile()


_graph = build_graph()


def run_query(question: str, session_as_of: date, conn: Any) -> dict[str, Any]:
    initial: QueryState = {
        "raw_query": question,
        "session_as_of": session_as_of,
        "subqueries": [],
        "all_hits": [],
        "all_results": [],
        "query_type": "simple",
        "wall_clock_start": _orch.time.monotonic(),  # uses gated_orchestrator's time — mock-compatible
        "facts": None,
        "missing_facts": [],
        "issue_queries": [],
        "interrupted": False,
        "interrupt_prompt": None,
        "_pending_results": [],
        "_response": {},
    }
    result = _graph.invoke(
        initial,
        config={"configurable": {"conn": conn}, "recursion_limit": 10},
    )
    return result["_response"]
```

**Key change:** `wall_clock_start: _orch.time.monotonic()` — accesses `time.monotonic` through the `gated_orchestrator` module object so that test monkeypatches on `app.retrieval.gated_orchestrator.time.monotonic` are seen here. This replaces the `_graph_clock` hack entirely.

---

### 3. `tests/test_orchestrator.py` — update + add tests

The 3 tests that previously mocked `_classify_and_decompose` must now mock `_fact_extract`. The mock return format changes from:

```python
# OLD — classify_and_decompose
[{"subquery": question, "as_of": as_of}]

# NEW — _fact_extract
{
    "facts": None,
    "missing_facts": [],
    "issue_queries": [{"query": question, "as_of": as_of, "work_type_hint": None}],
}
```

**Test 4 (`test_classifier_failure_falls_back_to_simple`) is unchanged** — it calls `orchestrator._classify_and_decompose(...)` directly and that function still exists.

**Tests 5 and 6** (graph compile + schema) may be updated if their assertions need adjusting (e.g., test 5 currently mocks `_classify_and_decompose` — update to `_fact_extract`).

Here are the updated/new tests. Replace the file content below the `_validating_gate` helper with these:

```python
def test_simple_query_uses_single_session_as_of(monkeypatch: Any) -> None:
    seen: list[date] = []

    def retrieve(conn: object, query: str, as_of: date, k: int = 5) -> list[dict[str, str]]:
        seen.append(as_of)
        return [{"component_uri": "/law/1", "text_ne": "text"}]

    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [{"query": question, "as_of": as_of, "work_type_hint": None}],
        },
    )
    monkeypatch.setattr(orchestrator, "retrieve_postgres", retrieve)
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {"claims": [{"claim": "ok", "evidence_id": "/law/1"}]},
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "simple"
    assert seen == [date(2024, 1, 1)]
    assert body["results"][0]["as_of"] == "2024-01-01"


def test_complex_query_validates_each_subquery_as_of(monkeypatch: Any) -> None:
    validate_as_ofs: list[date] = []
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
            "facts": {"parties": [], "events": [], "dates": [], "location": None},
            "missing_facts": [],
            "issue_queries": [
                {"query": "old", "as_of": date(2020, 1, 1), "work_type_hint": None},
                {"query": "new", "as_of": date(2024, 1, 1), "work_type_hint": None},
            ],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {
            "claims": [{"claim": question, "evidence_id": hits[0]["component_uri"]}]
        },
    )

    def validate(
        claims: list[dict[str, str]], as_of: date, conn: object
    ) -> list[dict[str, Any]]:
        validate_as_ofs.append(as_of)
        return _validating_gate(claims, as_of, conn)

    monkeypatch.setattr(orchestrator, "validate_and_render", validate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "complex"
    assert validate_as_ofs == [date(2020, 1, 1), date(2024, 1, 1)]
    assert [r["as_of"] for r in body["results"]] == ["2020-01-01", "2024-01-01"]


def test_wall_clock_cap_returns_validated_so_far(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": "first", "as_of": as_of, "work_type_hint": None},
                {"query": "second", "as_of": as_of, "work_type_hint": None},
            ],
        },
    )
    times = iter([0.0, 0.0, orchestrator.WALL_CLOCK_CAP + 0.1])
    monkeypatch.setattr(
        "app.retrieval.gated_orchestrator.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {
            "claims": [{"claim": question, "evidence_id": hits[0]["component_uri"]}]
        },
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert [r["claim"] for r in body["results"]] == ["first"]


def test_classifier_failure_falls_back_to_simple(monkeypatch: Any) -> None:
    import langchain_openai

    class FailingChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
            raise RuntimeError("down")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FailingChatOpenAI)

    subqueries = orchestrator._classify_and_decompose("q", date(2024, 1, 1))

    assert subqueries == [{"subquery": "q", "as_of": date(2024, 1, 1)}]


def test_graph_compiles_and_returns_expected_shape(monkeypatch: Any) -> None:
    """Graph wires correctly and answer() returns the right response shape."""
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [{"query": question, "as_of": as_of, "work_type_hint": None}],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": "/law/1", "text_ne": "text"}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {"claims": [{"claim": "ok", "evidence_id": "/law/1"}]},
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    body = orchestrator.answer("q", date(2024, 1, 1), object())

    assert "as_of" in body
    assert "query_type" in body
    assert "abstained" in body
    assert "results" in body
    assert body["results"][0]["claim"] == "ok"


def test_query_state_schema_complete() -> None:
    """QueryState TypedDict has all required Stage 1+ fields."""
    import typing

    from app.retrieval.query_state import QueryState

    keys = set(typing.get_type_hints(QueryState).keys())
    required = {
        "raw_query", "session_as_of", "subqueries", "all_hits",
        "all_results", "query_type", "wall_clock_start",
        "facts", "missing_facts", "issue_queries",
        "interrupted", "interrupt_prompt", "_pending_results",
    }
    assert required.issubset(keys)


def test_fact_extract_success_populates_issue_queries(monkeypatch: Any) -> None:
    """_fact_extract parses Gemini response and returns normalised issue_queries."""
    import json

    class FakeResp:
        content = json.dumps({
            "facts": {"parties": ["landlord"], "events": ["eviction"], "dates": [], "location": None},
            "missing_facts": [{"fact": "written agreement?", "type": "clarifying"}],
            "issue_queries": [
                {"query": "भाडा सम्झौता सम्बन्धी कानून", "as_of": "2024-01-01", "work_type_hint": "Act"},
                {"query": "घर खाली गराउने प्रक्रिया", "as_of": None, "work_type_hint": None},
            ],
        })

    import langchain_google_genai

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeLLM)
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            GEMINI_API_KEY="key",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )

    result = orchestrator._fact_extract("eviction question", date(2024, 1, 1))

    assert result["facts"]["parties"] == ["landlord"]
    assert result["missing_facts"][0]["type"] == "clarifying"
    assert len(result["issue_queries"]) == 2
    assert result["issue_queries"][0]["as_of"] == date(2024, 1, 1)
    assert result["issue_queries"][1]["as_of"] == date(2024, 1, 1)  # null → session_as_of


def test_fact_extract_failure_returns_single_query_fallback(monkeypatch: Any) -> None:
    """_fact_extract falls back to single raw query when Gemini is unavailable."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(GEMINI_API_KEY="", LANGFUSE_PUBLIC_KEY=""),
    )

    result = orchestrator._fact_extract("what is the notice period?", date(2024, 6, 1))

    assert result["facts"] is None
    assert result["missing_facts"] == []
    assert len(result["issue_queries"]) == 1
    assert result["issue_queries"][0]["query"] == "what is the notice period?"
    assert result["issue_queries"][0]["as_of"] == date(2024, 6, 1)
```

---

## Why monkeypatching still works

`retrieve_generate_node` calls `_orch.retrieve_postgres(...)`, `_orch._model_claims(...)` etc. — all via the module object. Tests patch `orchestrator.retrieve_postgres`, `orchestrator._model_claims` etc. These are the same module, so patches are seen at call time. No change from Stage 1.

`fact_extractor_node` calls `_orch._fact_extract(...)`. Tests patch `orchestrator._fact_extract` (note: `orchestrator` is `gated_orchestrator`, which is where `_fact_extract` lives). ✓

`run_query` sets `wall_clock_start = _orch.time.monotonic()`. When test patches `"app.retrieval.gated_orchestrator.time.monotonic"`, that patch replaces `time.monotonic` in `gated_orchestrator`'s `time` module attribute. `_orch.time.monotonic` reads this attribute at call time. ✓

---

## Wall-clock mock call sequence (Stage 2)

For `test_wall_clock_cap_returns_validated_so_far`, the 3 monotonic calls happen in this order:
1. `run_query` → `_orch.time.monotonic()` → 0.0 (wall_clock_start)
2. First iteration of issue_queries loop → `_wall_clock_expired(0.0)` → `time.monotonic()` → 0.0 (not expired)
3. Second iteration → `_wall_clock_expired(0.0)` → `time.monotonic()` → 20.1 (expired, break)

`_fact_extract` is mocked → no additional monotonic calls from Gemini internals.

---

## Fact extractor output format

```python
{
    "facts": {
        "parties": list[str],
        "events": list[str],
        "dates": list[str],
        "location": str | None,
    } | None,
    "missing_facts": [
        {"fact": str, "type": "required" | "clarifying" | "informational"}
    ],
    "issue_queries": [
        {"query": str, "as_of": date, "work_type_hint": str | None}
    ],
}
```

`issue_queries` must always have at least one entry (fallback ensures this).
`issue_queries[n]["as_of"]` is always a `date` object (never a string) after parsing.
`_pending_results[n]["as_of"]` continues to be a `date` object (so `validate_node` is unchanged).

---

## System design refs

- ADR-001 §Node 1 — Fact Extractor, §Node 2 — Parallel Retriever
- `system-design.md` §2 Core Invariants (unchanged), §8 query plane, §14 PS-6 (eligibility gate on every retrieval branch — still enforced inside `retrieve_postgres`)
- `system-design.md` §14 PS-8 — Romanized Nepali: fact extractor writes `issue_queries` in Devanagari Nepali (instruction in prompt). This strengthens PS-8 compliance.

---

## Zero-tolerance gates

- `repealed-as-current = 0` — unchanged. No temporal eligibility logic touched.
- `not-yet-effective-as-current = 0` — unchanged.
- Eligibility gate runs inside `retrieve_postgres` on every issue branch.

---

## Required checks

```bash
make test    # must show 57 passed (55 + 2 new)
make lint
```

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No AI attribution. No `Co-Authored-By` trailers. No "Generated with Claude" lines.

---

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions.
