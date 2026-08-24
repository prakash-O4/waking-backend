# Task: AGENT-1 — LangGraph skeleton (Stage 1)

**Branch:** `agent/stage-1-skeleton`
**Base:** `dev`
**Engineer:** Pi
**ADR:** `docs/adr-001-multi-agent-query-architecture.md`

---

## Objective

Wire the existing linear query pipeline into a LangGraph graph.
**No behaviour change.** Identical inputs, identical outputs, identical degraded-mode behaviour.
This stage exists to prove the graph compiles, runs, and enforces caps — not to add new agents.

Stage 2 (Fact Extractor + parallel retrieval) is a separate task and must not be started here.

---

## Acceptance criteria

1. `orchestrator.answer(question, session_as_of, conn)` signature and return format are unchanged.
2. All 4 existing tests in `tests/test_orchestrator.py` pass without any modification to those tests.
3. The LangGraph graph compiles at import time (no deferred build errors).
4. Wall-clock cap (`WALL_CLOCK_CAP = 20s`) is still enforced — existing test proves this.
5. Two new tests added covering: graph compiles and invokes, `QueryState` schema validates.
6. `make test` green (53 → 55 passing), `make lint` clean.

---

## Exact scope

**New files:**
- `app/retrieval/query_state.py`
- `app/retrieval/query_graph.py`

**Modified files:**
- `requirements.txt` — add `langgraph>=1.2`
- `app/retrieval/gated_orchestrator.py` — `answer()` delegates to graph; all helper functions remain
- `tests/test_orchestrator.py` — add 2 new tests only; do not touch existing 4

Do NOT modify any other file. Do NOT add new agents, nodes beyond the 4 below, or new LLM calls.

---

## Implementation guide

### 1. `requirements.txt`

Add one line near `langchain` packages:
```
langgraph>=1.2
```

---

### 2. `app/retrieval/query_state.py`

```python
from __future__ import annotations

from datetime import date
from typing import Any
from typing_extensions import TypedDict


class QueryState(TypedDict):
    # ── active in Stage 1 ──────────────────────────────────────
    raw_query: str
    session_as_of: date
    subqueries: list[dict[str, Any]]
    all_hits: list[dict[str, Any]]
    all_results: list[dict[str, Any]]
    query_type: str
    wall_clock_start: float

    # ── placeholder for Stage 2+ (None / empty in Stage 1) ────
    facts: Any                        # StructuredFacts in Stage 2
    missing_facts: list[dict[str, Any]]
    issue_queries: list[dict[str, Any]]
    interrupted: bool
    interrupt_prompt: str | None
```

---

### 3. `app/retrieval/query_graph.py`

```python
from __future__ import annotations

import time
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import app.retrieval.gated_orchestrator as _orch
from app.retrieval.query_state import QueryState


# ── node functions ─────────────────────────────────────────────────────────

def classify_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    subqueries = _orch._classify_and_decompose(
        state["raw_query"], state["session_as_of"]
    )
    return {
        "subqueries": subqueries,
        "query_type": "simple" if len(subqueries) == 1 else "complex",
    }


def retrieve_generate_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    all_hits: list[dict[str, Any]] = []
    partial_results: list[dict[str, Any]] = []
    query_type = state["query_type"]

    for subquery in state["subqueries"]:
        if _orch._wall_clock_expired(state["wall_clock_start"]):
            break

        subquery_text: str = subquery["subquery"]
        subquery_as_of: date = subquery["as_of"]

        hits = _orch.retrieve_postgres(conn, subquery_text, subquery_as_of, k=5)
        all_hits.extend(hits)
        if not hits:
            continue

        parsed = _orch._model_claims(subquery_text, hits)
        if parsed is None:
            claims = _orch._extractive_claim(hits)
            query_type = "extractive"
        elif parsed.get("abstain") or not parsed.get("claims"):
            continue
        else:
            claims = parsed["claims"]

        partial_results.append(
            {"claims": claims, "as_of": subquery_as_of}
        )

    return {
        "all_hits": all_hits,
        "query_type": query_type,
        "_pending_results": partial_results,   # internal key, consumed by validate_node
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
        raw_query, session_as_of, query_type, all_results, all_hits,
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


# ── graph ──────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve_generate", retrieve_generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("assemble", assemble_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve_generate")
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
        "wall_clock_start": time.monotonic(),
        "facts": None,
        "missing_facts": [],
        "issue_queries": [],
        "interrupted": False,
        "interrupt_prompt": None,
    }
    result = _graph.invoke(
        initial,
        config={"configurable": {"conn": conn}, "recursion_limit": 10},
    )
    return result["_response"]
```

**Critical note on `_pending_results`:** LangGraph state keys must be declared in the TypedDict. Add `_pending_results: list[dict[str, Any]]` to `QueryState` so the graph can pass it between nodes. Set it to `[]` in the initial state.

---

### 4. `app/retrieval/gated_orchestrator.py` — changes only

#### 4a. Extract the Langfuse emit into a standalone helper

The current `answer()` has inline Langfuse trace emission. Extract it so `assemble_node` can call it:

```python
def _emit_answer_trace_from_state(
    raw_query: str,
    session_as_of: date,
    query_type: str,
    all_results: list[dict[str, Any]],
    all_hits: list[dict[str, Any]],
    wall_clock_start: float,
) -> None:
    """Extracted from answer() for graph node use."""
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return
    try:
        __import__("langfuse")
    except ImportError:
        return
    claims_passed = sum(1 for r in all_results if not r.get("abstained"))
    claims_abstained = sum(1 for r in all_results if r.get("abstained"))
    top_chunk_scores = sorted(
        [hit.get("score", 0.0) for hit in all_hits], reverse=True
    )[:5]
    _emit_answer_trace({
        "query_hash": hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
        "as_of": session_as_of.isoformat(),
        "query_type": query_type,
        "latency_ms": _elapsed_ms(wall_clock_start),
        "gate_decision": "abstained" if not all_results else "answered",
        "result_count": len(all_results),
        "top_chunk_scores": top_chunk_scores,
        "validation_claims_passed": claims_passed,
        "validation_claims_abstained": claims_abstained,
    })
```

#### 4b. Replace `answer()` body

```python
def answer(question: str, session_as_of: date, conn: connection) -> dict[str, Any]:
    from app.retrieval.query_graph import run_query
    return run_query(question, session_as_of, conn)
```

Everything else in `gated_orchestrator.py` — `_classify_and_decompose`, `_model_claims`,
`_extractive_claim`, `_wall_clock_expired`, `_emit_answer_trace`, `_timing_start`,
`_elapsed_ms`, `_langfuse_callback`, `WALL_CLOCK_CAP`, `MAX_SUBQUERIES`,
`EXTRACTIVE_CHARS` — stays unchanged and at module level. This is mandatory for
test monkeypatching to keep working.

---

## Why monkeypatching still works

The existing tests do:
```python
monkeypatch.setattr(orchestrator, "_classify_and_decompose", ...)
monkeypatch.setattr(orchestrator, "retrieve_postgres", ...)
```

`query_graph.py` imports via `import app.retrieval.gated_orchestrator as _orch` and calls
`_orch._classify_and_decompose(...)`. Because `_orch` is the same module object as
`orchestrator` in the tests, monkeypatching the module attribute is seen by the graph
nodes at call time. Do NOT use `from app.retrieval.gated_orchestrator import _classify_and_decompose`
in `query_graph.py` — that would snapshot the function and bypass monkeypatching.

---

## New tests (`tests/test_orchestrator.py`)

Add these two tests at the bottom of the existing file. Do not touch the existing 4.

```python
def test_graph_compiles_and_returns_expected_shape(monkeypatch: Any) -> None:
    """Graph wires correctly and answer() returns the right response shape."""
    monkeypatch.setattr(
        orchestrator,
        "_classify_and_decompose",
        lambda question, as_of: [{"subquery": question, "as_of": as_of}],
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
    """QueryState TypedDict has all required Stage 1 fields."""
    from app.retrieval.query_state import QueryState
    import typing

    keys = set(typing.get_type_hints(QueryState).keys())
    required = {
        "raw_query", "session_as_of", "subqueries", "all_hits",
        "all_results", "query_type", "wall_clock_start",
        "facts", "missing_facts", "issue_queries",
        "interrupted", "interrupt_prompt", "_pending_results",
    }
    assert required.issubset(keys)
```

---

## System Design refs

- ADR-001 §Stage 1 — LangGraph skeleton
- `system-design.md` §8 — query plane; this change wires the existing plane into a graph
- No PS requirements are in scope; gates are called from inside the same node functions

## Zero-tolerance gates

Unaffected. No eligibility or validation gate logic changes.

---

## Required checks

```bash
make test    # must show 55 passed (53 + 2 new)
make lint
```

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No AI attribution. No `Co-Authored-By` trailers.

---

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions.
