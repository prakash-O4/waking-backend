from __future__ import annotations

import hashlib
import time
import unicodedata
from datetime import date
from typing import Any, cast

from openai import AzureOpenAI
from psycopg2.extensions import connection

from app.config import azure_base_url, get_settings
from app.retrieval.eligibility_gate import eligible_chunk_ids
from app.retrieval.reranker import rerank

_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
_RELEVANCE_THRESHOLD = 0.005
_lf_client: Any = None


def _get_lf_client() -> Any | None:
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return None
    global _lf_client
    if _lf_client is None:
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError:
            return None
        s = get_settings()
        _lf_client = Langfuse(
            public_key=s.LANGFUSE_PUBLIC_KEY,
            secret_key=s.LANGFUSE_SECRET_KEY,
            host=s.LANGFUSE_HOST,
        )
    return _lf_client


def _span(trace: Any, stage: str, **metadata: Any) -> None:
    if trace is not None:
        trace.span(name=f"stage.{stage}", metadata=metadata)


def _preprocess(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_DIGIT_MAP)


def _embed_query(text: str) -> list[float]:
    s = get_settings()
    client = AzureOpenAI(
        api_key=s.AZURE_OPENAI_KEY,
        azure_endpoint=azure_base_url(),
        api_version=s.AZURE_OPENAI_API_VERSION,
    )
    resp = client.embeddings.create(
        model=s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text,
        dimensions=s.AZURE_OPENAI_EMBEDDING_DIMENSIONS,
    )
    return cast(list[float], resp.data[0].embedding)


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _hit(row: tuple[Any, ...], score: float) -> dict[str, Any]:
    chunk_id, text, text_hash, act_name, case_id, chunk_type, section_number = row[:7]
    source_id = row[7] if len(row) == 8 else ""
    return {
        "component_uri": str(chunk_id),
        "text_ne": text,
        "text_hash": text_hash,
        "score": float(score),
        "work_title_ne": act_name or case_id or "",
        "chunk_type": chunk_type,
        "section_number": section_number or "",
        "document_source_id": str(source_id) if source_id else "",
    }


def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    query = _preprocess(query)
    lf = _get_lf_client()
    trace = (
        lf.trace(
            name="rag.retrieval",
            metadata={
                "query_hash": hashlib.sha256(query.encode()).hexdigest(),
                "as_of": str(as_of),
                "k": k,
            },
        )
        if lf
        else None
    )

    t0 = time.monotonic()
    eligible = list(eligible_chunk_ids(conn, as_of))
    _span(
        trace,
        "eligibility_gate",
        eligible_count=len(eligible),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    if not eligible:
        if lf:
            lf.flush()
        return []

    qvec = _embed_query(query)
    limit = k * 3
    with conn.cursor() as cur:
        t0 = time.monotonic()
        cur.execute(
            """
            SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
                   c.chunk_type, c.section_number, d.source_id,
                   1 - (c.embedding <=> %(qvec)s::vector) AS vec_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id::text = ANY(%(eligible)s)
            ORDER BY c.embedding <=> %(qvec)s::vector
            LIMIT %(limit)s
            """,
            {"qvec": qvec, "eligible": eligible, "limit": limit},
        )
        vector_rows = cur.fetchall()
        _span(
            trace,
            "vector_search",
            candidate_count=len(vector_rows),
            top_score=float(vector_rows[0][8 if len(vector_rows[0]) > 8 else 7])
            if vector_rows
            else 0.0,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        lexical_rows: list[tuple[Any, ...]] = []
        lexical_ran = any(ch.isalpha() for ch in query)
        t0 = time.monotonic()
        if lexical_ran:
            cur.execute(
                """
                SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
                       c.chunk_type, c.section_number, d.source_id,
                       ts_rank_cd(to_tsvector('simple', c.chunk_text),
                                  plainto_tsquery('simple', %(query)s)) AS lex_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id::text = ANY(%(eligible)s)
                  AND to_tsvector('simple', c.chunk_text) @@ plainto_tsquery('simple', %(query)s)
                LIMIT %(limit)s
                """,
                {"query": query, "eligible": eligible, "limit": limit},
            )
            lexical_rows = cur.fetchall()
        _span(
            trace,
            "lexical_search",
            ran=lexical_ran,
            candidate_count=len(lexical_rows),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    rows = {str(row[0]): row for row in [*vector_rows, *lexical_rows]}
    t0 = time.monotonic()
    rrf_scores = _rrf(
        [[str(r[0]) for r in vector_rows], [str(r[0]) for r in lexical_rows]]
    )
    _span(
        trace,
        "rrf_fusion",
        merged_count=len(rrf_scores),
        top_rrf_score=rrf_scores[0][1] if rrf_scores else 0.0,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )

    t0 = time.monotonic()
    candidates = [
        _hit(rows[chunk_id], score)
        for chunk_id, score in rrf_scores
        if score >= _RELEVANCE_THRESHOLD
    ][: k * 2]
    _span(
        trace,
        "relevance_gate",
        passed_count=len(candidates),
        abstained=not candidates,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    if not candidates:
        if lf:
            lf.flush()
        return []

    t0 = time.monotonic()
    ranked = rerank(query, candidates, k)
    _span(
        trace,
        "rerank",
        ran=bool(get_settings().COHERE_API_KEY),
        final_count=len(ranked),
        top_score=float(ranked[0].get("score", 0.0)) if ranked else 0.0,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )

    final_ids = [h["component_uri"] for h in ranked]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
                   c.chunk_type, c.section_number, d.source_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id::text = ANY(%(ids)s)
            """,
            {"ids": final_ids},
        )
        full_rows = {str(row[0]): row for row in cur.fetchall()}

    if lf:
        lf.flush()
    return [
        _hit(full_rows[h["component_uri"]], h["score"])
        for h in ranked
        if h["component_uri"] in full_rows
    ]
