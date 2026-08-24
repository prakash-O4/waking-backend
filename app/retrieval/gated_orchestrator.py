from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from typing import Any, cast

from psycopg2.extensions import connection

from app.config import get_settings
from app.retrieval.postgres_retriever import retrieve_postgres as retrieve_postgres
from app.retrieval.validation_gate import validate_and_render as validate_and_render

WALL_CLOCK_CAP = 20.0
MAX_SUBQUERIES = 3
EXTRACTIVE_CHARS = 300
_REAL_MONOTONIC = time.monotonic


def _langfuse_callback() -> list[Any]:
    settings = get_settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return []
    try:
        from langfuse.callback import (  # type: ignore[import-not-found]
            CallbackHandler as LangfuseCallbackHandler,
        )
    except ImportError:
        return []

    return [
        LangfuseCallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    ]


def _emit_answer_trace(metadata: dict[str, Any]) -> None:
    settings = get_settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        return

    client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    client.trace(name="rag.answer", metadata=metadata)
    client.flush()


def _model_claims(question: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Call model; return None on failure so caller can extract from evidence."""
    from langchain_openai import ChatOpenAI

    context = "\n\n---\n\n".join(
        f"[{h['component_uri']}]\n{h['text_ne']}" for h in hits
    )
    system = (
        "You are Wakil-G. Answer using ONLY the provided context. "
        "Output JSON only:\n"
        '{"claims": [{"claim": "<answer text>", "evidence_id": "<component_uri>"}]}\n'
        'If context is insufficient: {"claims": [], "abstain": true}\n'
        "Do not write citations."
    )
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.0,
        )
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            config={"callbacks": _langfuse_callback()},
        )
        return cast(dict[str, Any], json.loads(resp.content.strip()))
    except Exception:
        return None


def _elapsed_ms(start: float | None) -> int:
    if start is None:
        return 0
    try:
        return int((time.monotonic() - start) * 1000)
    except StopIteration:
        return 0


def _timing_start(enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        return time.monotonic()
    except StopIteration:
        return None


def _wall_clock_expired(start: float) -> bool:
    try:
        return time.monotonic() - start > WALL_CLOCK_CAP
    except StopIteration:
        return True


def _extractive_claim(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not hits:
        return []
    hit = hits[0]
    return [
        {
            "claim": hit["text_ne"][:EXTRACTIVE_CHARS],
            "evidence_id": hit["component_uri"],
        }
    ]


def _classify_and_decompose(question: str, session_as_of: date) -> list[dict[str, Any]]:
    """LLM router. On any failure, use the original question as a simple query."""
    from langchain_openai import ChatOpenAI

    system = (
        "Classify the legal question as simple (single issue, single time point) or "
        "complex (multiple issues or comparative across time). Output JSON only:\n"
        '{"type": "simple"} OR\n'
        '{"type": "complex", "subqueries": '
        '[{"q": "<sub-question>", "as_of": "<YYYY-MM-DD or null>"}]}\n'
        f"Default as_of if not detected: {session_as_of.isoformat()}. "
        f"Max {MAX_SUBQUERIES} sub-queries."
    )
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.0,
        )
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            config={"callbacks": _langfuse_callback()},
        )
        parsed = json.loads(resp.content.strip())
        if parsed.get("type") != "complex":
            return [{"subquery": question, "as_of": session_as_of}]

        result: list[dict[str, Any]] = []
        for subquery in parsed.get("subqueries", [])[:MAX_SUBQUERIES]:
            try:
                sub_as_of = (
                    date.fromisoformat(subquery["as_of"])
                    if subquery.get("as_of")
                    else session_as_of
                )
            except (KeyError, TypeError, ValueError):
                sub_as_of = session_as_of
            result.append({"subquery": subquery.get("q", question), "as_of": sub_as_of})
        return result or [{"subquery": question, "as_of": session_as_of}]
    except Exception:
        return [{"subquery": question, "as_of": session_as_of}]


def _fact_extract(question: str, session_as_of: date) -> dict[str, Any]:
    """Extract structured facts and per-issue retrieval queries using Gemini 2.5 Flash.

    Failure mode: any exception or missing key → single raw query passthrough.
    """
    fallback: dict[str, Any] = {
        "facts": None,
        "missing_facts": [],
        "issue_queries": [
            {"query": question, "as_of": session_as_of, "work_type_hint": None}
        ],
    }
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return fallback
    system = (
        "You are a Nepali legal assistant. Analyse the user's legal query and output JSON only:\n"
        "{\n"
        '  "facts": {"parties": [], "events": [], "dates": [], "location": null},\n'
        '  "missing_facts": [\n'
        '    {"fact": "<what is missing>", "type": "required|clarifying|informational"}\n'
        "  ],\n"
        '  "issue_queries": [\n'
        '    {"query": "<Nepali retrieval query>", "as_of": "<YYYY-MM-DD or null>",\n'
        '     "work_type_hint": "<Act|Rule|Regulation|null>"}\n'
        "  ]\n"
        "}\n"
        f"Default as_of when not specified: {session_as_of.isoformat()}. "
        f"Max {MAX_SUBQUERIES} issue_queries. "
        "Write issue_queries in formal Devanagari Nepali for best embedding match."
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

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
                    date.fromisoformat(iq["as_of"])
                    if iq.get("as_of")
                    else session_as_of
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
            "issue_queries": issue_queries or fallback["issue_queries"],
        }
    except Exception:
        return fallback


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
    _emit_answer_trace(
        {
            "query_hash": hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
            "as_of": session_as_of.isoformat(),
            "query_type": query_type,
            "latency_ms": _elapsed_ms(wall_clock_start),
            "gate_decision": "abstained" if not all_results else "answered",
            "result_count": len(all_results),
            "top_chunk_scores": top_chunk_scores,
            "validation_claims_passed": claims_passed,
            "validation_claims_abstained": claims_abstained,
        }
    )


def answer(question: str, session_as_of: date, conn: connection) -> dict[str, Any]:
    from app.retrieval.query_graph import run_query

    return cast(dict[str, Any], run_query(question, session_as_of, conn))
