from __future__ import annotations

import hashlib as _hashlib
from datetime import date, datetime
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import app.retrieval.gated_orchestrator as _orch
from app.retrieval.postgres_retriever import get_lf_client as _get_lf_client
from app.retrieval.query_state import QueryState


# ── nodes ──────────────────────────────────────────────────────────────────────


def fact_extractor_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    lf_trace = config["configurable"].get("lf_trace")
    result = _orch._fact_extract(state["raw_query"], state["session_as_of"], lf_trace=lf_trace)
    issue_queries = result["issue_queries"]
    missing_facts = result["missing_facts"]

    required = [mf for mf in missing_facts if mf.get("type") == "required"]
    interrupted = bool(required)
    interrupt_prompt: str | None = None
    if required:
        interrupt_prompt = "To answer your question I need to know: " + "; ".join(
            mf.get("fact", "") for mf in required
        )

    return {
        "facts": result["facts"],
        "missing_facts": missing_facts,
        "issue_queries": issue_queries,
        "query_type": "simple" if len(issue_queries) == 1 else "complex",
        "interrupted": interrupted,
        "interrupt_prompt": interrupt_prompt,
    }


def retrieve_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    """Pure retrieval — no LLM calls. Tags each hit with _issue_idx."""
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    all_hits: list[dict[str, Any]] = []

    issue_queries = state["issue_queries"] or [
        {
            "query": state["raw_query"],
            "as_of": state["session_as_of"],
            "work_type_hint": None,
        }
    ]
    for idx, iq in enumerate(issue_queries):
        if _orch._wall_clock_expired(state["wall_clock_start"]):
            break
        hits = _orch.retrieve_postgres(conn, iq["query"], iq["as_of"], k=5, lf_trace=lf_trace)
        for h in hits:
            all_hits.append({**h, "_issue_idx": idx})

    return {"all_hits": all_hits, "_pending_results": []}


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


def authority_ranker_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    hits_in = len(state["all_hits"])
    ranked = _orch._authority_rank_hits(state["all_hits"], conn)
    if lf_trace is not None:
        try:
            sp = lf_trace.span(name="authority_ranking")
            sp.end(metadata={"hits_in": hits_in, "hits_out": len(ranked)})
        except Exception:
            pass
    return {"all_hits": ranked}


def cross_ref_resolver_node(
    state: QueryState, config: RunnableConfig
) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    additional = _orch._resolve_cross_refs(
        state["all_hits"], state["session_as_of"], conn
    )
    if lf_trace is not None:
        try:
            sp = lf_trace.span(name="cross_ref_resolution")
            sp.end(metadata={"cross_refs_added": len(additional)})
        except Exception:
            pass
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}


def validate_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    all_results: list[dict[str, Any]] = []

    for pending in state.get("_pending_results", []):
        orig_claims = pending["claims"]
        validated = _orch.validate_and_render(orig_claims, pending["as_of"], conn)
        for i, result in enumerate(validated):
            result["as_of"] = pending["as_of"].isoformat()
            if i < len(orig_claims):
                for field in ("issue", "applicability", "condition"):
                    if field in orig_claims[i]:
                        result[field] = orig_claims[i][field]
        all_results.extend(validated)

    if lf_trace is not None:
        try:
            sp = lf_trace.span(name="validation")
            sp.end(
                metadata={
                    "claims_passed": sum(
                        1 for r in all_results if not r.get("abstained")
                    ),
                    "claims_abstained": sum(
                        1 for r in all_results if r.get("abstained")
                    ),
                }
            )
        except Exception:
            pass
    return {"all_results": all_results}


def answer_composer_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    lf_trace = config["configurable"].get("lf_trace")
    session_as_of = state["session_as_of"]
    query_type = state["query_type"]

    if state.get("interrupted"):
        if lf_trace is not None:
            try:
                lf_trace.update(
                    output={"interrupted": True, "result_count": 0},
                    end_time=datetime.now(),
                )
            except Exception:
                pass
        return {
            "_response": {
                "as_of": session_as_of.isoformat(),
                "query_type": query_type,
                "abstained": False,
                "results": [],
                "interrupted": True,
                "interrupt_prompt": state.get("interrupt_prompt"),
            }
        }

    all_results = state["all_results"]
    all_hits = state["all_hits"]
    raw_query = state["raw_query"]

    conflict_hits = [
        {
            "component_uri": h.get("component_uri", ""),
            "work_type": h.get("work_type", ""),
            "tier": h.get("tier"),
            "section_number": h.get("section_number", ""),
        }
        for h in all_hits
        if h.get("conflict_flag")
    ]

    composed = _orch._compose_answer(
        state["facts"],
        state["missing_facts"],
        all_results,
        conflict_hits,
        session_as_of,
        lf_trace=lf_trace,
    )

    _orch._emit_answer_trace_from_state(
        lf_trace,
        raw_query,
        session_as_of,
        query_type,
        all_results,
        all_hits,
        state["wall_clock_start"],
    )

    if composed is None:
        return {
            "_response": {
                "as_of": session_as_of.isoformat(),
                "query_type": query_type,
                "abstained": not all_results,
                "results": all_results,
            }
        }

    composed["query_type"] = query_type
    return {"_response": composed}


# ── graph ──────────────────────────────────────────────────────────────────────


def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("fact_extractor", fact_extractor_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("authority_ranker", authority_ranker_node)
    builder.add_node("cross_ref_resolver", cross_ref_resolver_node)
    builder.add_node("reasoner", reasoner_node)
    builder.add_node("validate", validate_node)
    builder.add_node("answer_composer", answer_composer_node)

    builder.add_edge(START, "fact_extractor")
    builder.add_conditional_edges(
        "fact_extractor",
        lambda state: "answer_composer" if state.get("interrupted") else "retrieve",
        {"answer_composer": "answer_composer", "retrieve": "retrieve"},
    )
    builder.add_edge("retrieve", "authority_ranker")
    builder.add_edge("authority_ranker", "cross_ref_resolver")
    builder.add_edge("cross_ref_resolver", "reasoner")
    builder.add_edge("reasoner", "validate")
    builder.add_edge("validate", "answer_composer")
    builder.add_edge("answer_composer", END)

    return builder.compile()


_graph = build_graph()


def run_query(question: str, session_as_of: date, conn: Any) -> dict[str, Any]:
    _lf = _get_lf_client()
    lf_trace = None
    if _lf is not None:
        try:
            s = _orch.get_settings()
            trace_input = (
                question
                if s.LANGFUSE_LOG_CONTENT
                else _hashlib.sha256(question.encode()).hexdigest()[:16]
            )
            lf_trace = _lf.trace(
                name="rag.query",
                input=trace_input,
                metadata={"as_of": session_as_of.isoformat()},
            )
        except Exception:
            pass

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
        config={
            "configurable": {"conn": conn, "lf_trace": lf_trace},
            "recursion_limit": 10,
        },
    )

    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            pass

    return cast(dict[str, Any], result["_response"])
