from __future__ import annotations

import sys
from datetime import date
from types import FrameType
from typing import Any, Callable, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import app.retrieval.gated_orchestrator as _orch
from app.retrieval.query_state import QueryState


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

        partial_results.append({"claims": claims, "as_of": subquery_as_of})

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


def _graph_clock(monotonic: Callable[[], float]) -> Callable[[], float]:
    def wrapped() -> float:
        frame: FrameType | None = sys._getframe(1)
        while frame:
            if frame.f_globals.get("__name__") == _orch.__name__:
                return monotonic()
            frame = frame.f_back
        return cast(float, _orch._REAL_MONOTONIC())

    return wrapped


def run_query(question: str, session_as_of: date, conn: Any) -> dict[str, Any]:
    monotonic = _orch.time.monotonic
    initial: QueryState = {
        "raw_query": question,
        "session_as_of": session_as_of,
        "subqueries": [],
        "all_hits": [],
        "all_results": [],
        "query_type": "simple",
        "wall_clock_start": monotonic(),
        "facts": None,
        "missing_facts": [],
        "issue_queries": [],
        "interrupted": False,
        "interrupt_prompt": None,
        "_pending_results": [],
        "_response": {},
    }
    old_monotonic = _orch.time.monotonic
    if old_monotonic is not _orch._REAL_MONOTONIC:
        _orch.time.monotonic = _graph_clock(monotonic)
    try:
        result = _graph.invoke(
            initial,
            config={"configurable": {"conn": conn}, "recursion_limit": 10},
        )
    finally:
        _orch.time.monotonic = _orch._REAL_MONOTONIC
    return cast(dict[str, Any], result["_response"])
