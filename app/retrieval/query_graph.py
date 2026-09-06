from __future__ import annotations

import hashlib as _hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Iterator, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import app.retrieval.gated_orchestrator as _orch
from app.retrieval.eligibility_gate import eligible_chunk_ids
from app.retrieval.postgres_retriever import get_lf_client as _get_lf_client
from app.retrieval.query_state import QueryState
from app.utils.loggers import logger

_ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")
MAX_RETRIEVER_FANOUT = 5


def _trace_error(
    lf_trace: Any, stage: str, error: Exception, *, fatal: bool = False
) -> None:
    msg = f"{stage} failed: {error}"
    logger.warning(msg)
    if lf_trace is None:
        return
    try:
        span = lf_trace.start_observation(
            name=f"error.{stage}", as_type="span", metadata={"error": msg}
        )
        span.end()
        if fatal:
            lf_trace.update(output={"error": msg})
            lf_trace.end()
    except Exception:
        pass


def _issue_queries_for_state(state: QueryState) -> list[dict[str, Any]]:
    return state.get("issue_queries") or [
        {
            "query": state.get("raw_query", ""),
            "as_of": state["session_as_of"],
            "work_type_hint": None,
        }
    ]


def _hit_as_of(state: QueryState, hit: dict[str, Any]) -> date:
    issue_queries = _issue_queries_for_state(state)
    idx = int(hit.get("_issue_idx", 0) or 0)
    if 0 <= idx < len(issue_queries):
        return cast(date, issue_queries[idx].get("as_of") or state["session_as_of"])
    return cast(date, state["session_as_of"])


def _degraded_mode(state: QueryState) -> list[str]:
    modes = {
        f"reranker_fallback:{h.get('reranker_tier')}"
        for h in state.get("all_hits", [])
        if h.get("reranker_tier") and h.get("reranker_tier") != "cohere"
    }
    if state.get("query_type") == "extractive":
        modes.add("reasoner_fallback:extractive")
    return sorted(modes)


# ── nodes ──────────────────────────────────────────────────────────────────────


def fact_extractor_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    result = _orch._fact_extract(
        state["raw_query"], state["session_as_of"], lf_trace=lf_trace
    )
    issue_queries = result["issue_queries"]
    missing_facts = list(result["missing_facts"])

    required = [mf for mf in missing_facts if mf.get("type") == "required"]
    interrupted = bool(required)
    if required and not _orch._wall_clock_expired(state["wall_clock_start"]):
        try:
            probe = _orch.retrieve_postgres(
                conn, state["raw_query"], state["session_as_of"], k=3, lf_trace=lf_trace
            )
        except Exception as e:
            _trace_error(lf_trace, "coverage_probe", e)
            probe = []
        if probe:
            interrupted = False
            missing_facts = [
                {**mf, "type": "clarifying"} if mf.get("type") == "required" else mf
                for mf in missing_facts
            ]
            required = []

    interrupt_prompt: str | None = None
    if interrupted:
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
    issue_queries = _issue_queries_for_state(state)

    def sequential() -> list[dict[str, Any]]:
        all_hits: list[dict[str, Any]] = []
        for idx, iq in enumerate(issue_queries):
            if _orch._wall_clock_expired(state["wall_clock_start"]):
                break
            hits = _orch.retrieve_postgres(
                conn, iq["query"], iq["as_of"], k=5, lf_trace=lf_trace
            )
            all_hits.extend({**h, "_issue_idx": idx} for h in hits)
        return all_hits

    if len(issue_queries) <= 1:
        return {"all_hits": sequential(), "_pending_results": []}

    pool = None
    try:
        from app.retrieval.db_pool import make_retrieval_pool

        max_workers = min(len(issue_queries), MAX_RETRIEVER_FANOUT)
        pool = make_retrieval_pool(max_workers)
        per_issue: list[list[dict[str, Any]]] = [[] for _ in issue_queries]

        def run_one(idx: int, iq: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
            if _orch._wall_clock_expired(state["wall_clock_start"]):
                return idx, []
            worker_conn = pool.getconn()
            try:
                # Langfuse trace objects are not assumed thread-safe; node-level traces stay on main thread.
                hits = _orch.retrieve_postgres(
                    worker_conn, iq["query"], iq["as_of"], k=5
                )
                return idx, [{**h, "_issue_idx": idx} for h in hits]
            finally:
                pool.putconn(worker_conn)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one, idx, iq)
                for idx, iq in enumerate(issue_queries)
            ]
            for future in futures:
                idx, hits = future.result()
                per_issue[idx] = hits
        return {
            "all_hits": [h for hits in per_issue for h in hits],
            "_pending_results": [],
        }
    except Exception as e:
        _trace_error(lf_trace, "parallel_retrieval", e)
        return {"all_hits": sequential(), "_pending_results": []}
    finally:
        if pool is not None:
            try:
                pool.closeall()
            except Exception:
                pass


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
            sp = lf_trace.start_observation(name="authority_ranking", as_type="span")
            sp.update(metadata={"hits_in": hits_in, "hits_out": len(ranked)})
            sp.end()
        except Exception:
            pass
    return {"all_hits": ranked}


def co_retrieve_parent_resolver_node(
    state: QueryState, config: RunnableConfig
) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    by_as_of: dict[date, list[dict[str, Any]]] = {}
    for hit in state["all_hits"]:
        by_as_of.setdefault(_hit_as_of(state, hit), []).append(hit)
    additional: list[dict[str, Any]] = []
    seen = {h.get("component_uri", "") for h in state["all_hits"]}
    for as_of, hits in by_as_of.items():
        for hit in _orch._resolve_co_retrieve_parents(hits, as_of, conn):
            if hit.get("component_uri") not in seen:
                seen.add(hit.get("component_uri"))
                additional.append(hit)
    if lf_trace is not None:
        try:
            sp = lf_trace.start_observation(
                name="co_retrieve_parent_resolution", as_type="span"
            )
            sp.update(metadata={"co_retrieve_parents_added": len(additional)})
            sp.end()
        except Exception:
            pass
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}


def cross_ref_resolver_node(
    state: QueryState, config: RunnableConfig
) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    by_as_of: dict[date, list[dict[str, Any]]] = {}
    for hit in state["all_hits"]:
        by_as_of.setdefault(_hit_as_of(state, hit), []).append(hit)
    additional: list[dict[str, Any]] = []
    seen = {h.get("component_uri", "") for h in state["all_hits"]}
    for as_of, hits in by_as_of.items():
        for hit in _orch._resolve_cross_refs(hits, as_of, conn):
            if hit.get("component_uri") not in seen:
                seen.add(hit.get("component_uri"))
                additional.append(hit)
    if lf_trace is not None:
        try:
            sp = lf_trace.start_observation(
                name="cross_ref_resolution", as_type="span"
            )
            sp.update(metadata={"cross_refs_added": len(additional)})
            sp.end()
        except Exception:
            pass
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}


def _fetch_enabling_chunk(
    conn: Any, hit: dict[str, Any], as_of: date
) -> dict[str, Any] | None:
    """Co-retrieve the enabling provision for a regulation chunk, if eligible."""
    chunk_id = hit.get("component_uri", "")
    if not chunk_id:
        return None

    with conn.cursor() as cur:
        cur.execute("SELECT work_id FROM chunks WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    subordinate_work_id = str(row[0])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT enabling_work_id, enabling_section_number, enabling_provision_type
            FROM work_relations
            WHERE subordinate_work_id = %(work_id)s
              AND enabling_work_id IS NOT NULL
              AND resolution_status IN ('auto_extracted', 'human_verified')
              AND valid_time @> %(as_of)s::timestamptz
            LIMIT 1
            """,
            {"work_id": subordinate_work_id, "as_of": as_of},
        )
        row = cur.fetchone()
    if not row:
        return None

    enabling_work_id, section_num_ascii, provision_type = row
    if not enabling_work_id or not section_num_ascii:
        return None

    section_num_deva = str(section_num_ascii).translate(_ASCII_TO_DEVA)
    # The indexer labels all law section chunks with a "दफा" prefix regardless of
    # whether the source uses "दफा" or "धारा"; match on section_number only.
    chunk_label = "धारा" if provision_type == "dhara" else "दफा"
    chunk_type = f"{chunk_label} {section_num_deva}"

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, d.source_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.work_id = %(work_id)s
              AND c.section_number = %(section)s
            ORDER BY c.chunk_index ASC
            LIMIT 1
            """,
            {"work_id": enabling_work_id, "section": section_num_deva},
        )
        row = cur.fetchone()
    if not row:
        return None

    enabling_chunk_id = str(row[0])
    eligible = eligible_chunk_ids(conn, as_of)
    if enabling_chunk_id not in eligible:
        return None

    return {
        "component_uri": enabling_chunk_id,
        "text_ne": row[1],
        "text_hash": row[2],
        "score": 0.0,
        "work_title_ne": row[3] or "",
        "chunk_type": chunk_type,
        "section_number": section_num_deva,
        "document_source_id": str(row[4]) if row[4] else "",
        "co_retrieved": True,
        "_issue_idx": hit.get("_issue_idx", 0),
    }


def enabling_power_resolver_node(
    state: QueryState, config: RunnableConfig
) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    hits = state.get("all_hits", [])
    additional: list[dict[str, Any]] = []

    existing_ids = {h.get("component_uri", "") for h in hits}
    try:
        for h in hits[:5]:
            if h.get("co_retrieved"):
                continue
            enabling = _fetch_enabling_chunk(conn, h, _hit_as_of(state, h))
            if enabling and enabling["component_uri"] not in existing_ids:
                existing_ids.add(enabling["component_uri"])
                additional.append(enabling)
    except Exception as e:
        # DB or eligibility-gate failure must not break the query path.
        _trace_error(lf_trace, "enabling_power_resolution", e)
        additional = []

    if lf_trace is not None:
        try:
            sp = lf_trace.start_observation(
                name="enabling_power_resolution", as_type="span"
            )
            sp.update(metadata={"enabling_chunks_added": len(additional)})
            sp.end()
        except Exception:
            pass
    if not additional:
        return {}
    return {"all_hits": hits + additional}


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
                for field in ("issue", "applicability", "condition", "quote"):
                    if field in orig_claims[i]:
                        result[field] = orig_claims[i][field]
        all_results.extend(validated)

    if lf_trace is not None:
        try:
            sp = lf_trace.start_observation(name="validation", as_type="span")
            sp.update(
                metadata={
                    "claims_passed": sum(
                        1 for r in all_results if not r.get("abstained")
                    ),
                    "claims_abstained": sum(
                        1 for r in all_results if r.get("abstained")
                    ),
                }
            )
            sp.end()
        except Exception:
            pass
    return {"all_results": all_results}


def answer_composer_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    lf_trace = config["configurable"].get("lf_trace")
    session_as_of = state["session_as_of"]
    query_type = state["query_type"]

    degraded_mode = _degraded_mode(state)

    if state.get("interrupted"):
        if lf_trace is not None:
            try:
                lf_trace.update(output={"interrupted": True, "result_count": 0})
                lf_trace.end()
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
                "degraded_mode": degraded_mode,
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
                "abstained": not any(not r.get("abstained") for r in all_results),
                "results": all_results,
                "degraded_mode": degraded_mode,
            }
        }

    composed = _orch._revalidate_composed(composed, all_results)
    composed["query_type"] = query_type
    composed["degraded_mode"] = degraded_mode
    return {"_response": composed}


# ── graph ──────────────────────────────────────────────────────────────────────


def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("fact_extractor", fact_extractor_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("authority_ranker", authority_ranker_node)
    builder.add_node("co_retrieve_parent_resolver", co_retrieve_parent_resolver_node)
    builder.add_node("cross_ref_resolver", cross_ref_resolver_node)
    builder.add_node("enabling_power_resolver", enabling_power_resolver_node)
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
    builder.add_edge("authority_ranker", "co_retrieve_parent_resolver")
    builder.add_edge("co_retrieve_parent_resolver", "cross_ref_resolver")
    builder.add_edge("cross_ref_resolver", "enabling_power_resolver")
    builder.add_edge("enabling_power_resolver", "reasoner")
    builder.add_edge("reasoner", "validate")
    builder.add_edge("validate", "answer_composer")
    builder.add_edge("answer_composer", END)

    return builder.compile()


_graph = build_graph()


def _start_trace(question: str, session_as_of: date) -> Any:
    _lf = _get_lf_client()
    if _lf is None:
        return None
    try:
        s = _orch.get_settings()
        trace_input = (
            question
            if s.LANGFUSE_LOG_CONTENT
            else _hashlib.sha256(question.encode()).hexdigest()[:16]
        )
        return _lf.start_observation(
            name="rag.query",
            as_type="span",
            input=trace_input,
            metadata={"as_of": session_as_of.isoformat()},
        )
    except Exception:
        return None


def _propagation_scope(user_id: str | None) -> Any:
    """Context manager propagating `user_id` to every observation created within it.

    Must wrap the root span's own creation (not just later children) — Langfuse
    aggregations by user_id only include observations created after this context
    is entered. Falls back to a no-op if langfuse isn't installed, matching the
    rest of this module's graceful-degradation-without-langfuse behavior.
    """
    try:
        from langfuse import propagate_attributes
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()
    return propagate_attributes(user_id=user_id)


def _initial_state(question: str, session_as_of: date) -> QueryState:
    return {
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


def run_query(
    question: str, session_as_of: date, conn: Any, user_id: str | None = None
) -> dict[str, Any]:
    with _propagation_scope(user_id):
        lf_trace = _start_trace(question, session_as_of)
        try:
            result = _graph.invoke(
                _initial_state(question, session_as_of),
                config={
                    "configurable": {"conn": conn, "lf_trace": lf_trace},
                    "recursion_limit": 10,
                },
            )
        except Exception as e:
            _trace_error(lf_trace, "query_pipeline", e, fatal=True)
            raise

    return cast(dict[str, Any], result["_response"])


def stream_query(
    question: str, session_as_of: date, conn: Any, user_id: str | None = None
) -> Iterator[dict[str, Any]]:
    with _propagation_scope(user_id):
        lf_trace = _start_trace(question, session_as_of)
        last = _orch.time.monotonic()
        final_sent = False
        try:
            for step in _graph.stream(
                _initial_state(question, session_as_of),
                config={
                    "configurable": {"conn": conn, "lf_trace": lf_trace},
                    "recursion_limit": 10,
                },
                stream_mode="updates",
            ):
                for stage in step:
                    if stage == "answer_composer":
                        response = step["answer_composer"]["_response"]
                        final_sent = True
                        yield {"stage": "final", "status": "done", "response": response}
                    else:
                        now = _orch.time.monotonic()
                        yield {
                            "stage": stage,
                            "status": "done",
                            "latency_ms": int((now - last) * 1000),
                        }
                        last = now
        except Exception as e:
            _trace_error(lf_trace, "stream_query_pipeline", e, fatal=True)
            if not final_sent:
                yield {"stage": "error", "status": "error", "detail": "query failed"}
