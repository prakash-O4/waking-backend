from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from typing import Any, cast

from psycopg2.extensions import connection

from app.config import Settings
from app.retrieval.dumb_retriever import retrieve as os_retrieve
from app.retrieval.postgres_retriever import retrieve_postgres
from app.retrieval.validation_gate import validate_and_render

WALL_CLOCK_CAP = 20.0
MAX_SUBQUERIES = 3
EXTRACTIVE_CHARS = 300


def _langfuse_callback() -> list[Any]:
    settings = Settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return []
    from langfuse.callback import (  # type: ignore[import-not-found]
        CallbackHandler as LangfuseCallbackHandler,
    )

    return [
        LangfuseCallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    ]


def _emit_answer_trace(metadata: dict[str, Any]) -> None:
    settings = Settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return
    from langfuse import Langfuse  # type: ignore[import-not-found]

    client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    client.trace(name="rag.answer", metadata=metadata)
    client.flush()


def _try_retrieve(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    """OpenSearch first; Postgres ILIKE fallback on OpenSearch connection errors."""
    try:
        return os_retrieve(query, as_of, k)
    except Exception as os_exc:
        import opensearchpy

        if isinstance(
            os_exc, (opensearchpy.ConnectionError, opensearchpy.TransportError)
        ):
            return retrieve_postgres(conn, query, as_of, k)
        raise


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


def answer(question: str, session_as_of: date, conn: connection) -> dict[str, Any]:
    start = time.monotonic()
    last_now = start
    subqueries = _classify_and_decompose(question, session_as_of)
    query_type = "simple" if len(subqueries) == 1 else "complex"
    all_results: list[dict[str, Any]] = []
    retrieved_uris: list[str] = []

    for subquery in subqueries:
        last_now = time.monotonic()
        if last_now - start > WALL_CLOCK_CAP:
            break

        subquery_text = cast(str, subquery["subquery"])
        subquery_as_of = cast(date, subquery["as_of"])
        hits = _try_retrieve(conn, subquery_text, subquery_as_of)
        retrieved_uris.extend(str(hit["component_uri"]) for hit in hits)
        if not hits:
            continue

        parsed = _model_claims(subquery_text, hits)
        if parsed is None:
            claims = _extractive_claim(hits)
            query_type = "extractive"
        elif parsed.get("abstain") or not parsed.get("claims"):
            continue
        else:
            claims = parsed["claims"]

        validated = validate_and_render(claims, subquery_as_of, conn)
        for result in validated:
            result["as_of"] = subquery_as_of.isoformat()
        all_results.extend(validated)

    response = {
        "as_of": session_as_of.isoformat(),
        "query_type": query_type,
        "abstained": not all_results,
        "results": all_results,
    }
    _emit_answer_trace(
        {
            "query_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "as_of": session_as_of.isoformat(),
            "query_type": query_type,
            "latency_ms": int((last_now - start) * 1000),
            "retrieved_uris": retrieved_uris,
            "gate_decision": "abstained" if not all_results else "answered",
            "result_count": len(all_results),
        }
    )
    return response
