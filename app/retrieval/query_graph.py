from __future__ import annotations

from datetime import date
from typing import Any, cast

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
        {
            "query": state["raw_query"],
            "as_of": state["session_as_of"],
            "work_type_hint": None,
        }
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


def authority_ranker_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    ranked = _orch._authority_rank_hits(state["all_hits"], conn)
    return {"all_hits": ranked}


def cross_ref_resolver_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    additional = _orch._resolve_cross_refs(
        state["all_hits"], state["session_as_of"], conn
    )
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}


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
    builder.add_node("authority_ranker", authority_ranker_node)
    builder.add_node("cross_ref_resolver", cross_ref_resolver_node)
    builder.add_node("validate", validate_node)
    builder.add_node("assemble", assemble_node)

    builder.add_edge(START, "fact_extractor")
    builder.add_edge("fact_extractor", "retrieve_generate")
    builder.add_edge("retrieve_generate", "authority_ranker")
    builder.add_edge("authority_ranker", "cross_ref_resolver")
    builder.add_edge("cross_ref_resolver", "validate")
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
        "wall_clock_start": _orch.time.monotonic(),
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
    return cast(dict[str, Any], result["_response"])
