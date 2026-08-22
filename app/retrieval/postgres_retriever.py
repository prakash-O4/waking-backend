from __future__ import annotations

from datetime import date
from typing import Any, cast
import unicodedata

from openai import AzureOpenAI
from psycopg2.extensions import connection

from app.config import azure_base_url, get_settings
from app.retrieval.eligibility_gate import eligible_chunk_ids
from app.retrieval.reranker import rerank

_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
_RELEVANCE_THRESHOLD = 0.005


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
    return {
        "component_uri": str(chunk_id),
        "text_ne": text,
        "text_hash": text_hash,
        "score": float(score),
        "work_title_ne": act_name or case_id or "",
        "chunk_type": chunk_type,
        "section_number": section_number or "",
    }


def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    query = _preprocess(query)
    eligible = list(eligible_chunk_ids(conn, as_of))
    if not eligible:
        return []

    qvec = _embed_query(query)
    limit = k * 3
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, chunk_text, span_sha256, act_name, case_id,
                   chunk_type, section_number,
                   1 - (embedding <=> %(qvec)s::vector) AS vec_score
            FROM chunks
            WHERE id::text = ANY(%(eligible)s)
            ORDER BY embedding <=> %(qvec)s::vector
            LIMIT %(limit)s
            """,
            {"qvec": qvec, "eligible": eligible, "limit": limit},
        )
        vector_rows = cur.fetchall()

        lexical_rows: list[tuple[Any, ...]] = []
        if any(ch.isalpha() for ch in query):
            cur.execute(
                """
                SELECT id::text, chunk_text, span_sha256, act_name, case_id,
                       chunk_type, section_number,
                       ts_rank_cd(to_tsvector('simple', chunk_text),
                                  plainto_tsquery('simple', %(query)s)) AS lex_score
                FROM chunks
                WHERE id::text = ANY(%(eligible)s)
                  AND to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', %(query)s)
                LIMIT %(limit)s
                """,
                {"query": query, "eligible": eligible, "limit": limit},
            )
            lexical_rows = cur.fetchall()

    rows = {str(row[0]): row for row in [*vector_rows, *lexical_rows]}
    scores = dict(
        _rrf([[str(r[0]) for r in vector_rows], [str(r[0]) for r in lexical_rows]])
    )
    candidates = [
        _hit(rows[chunk_id], score)
        for chunk_id, score in scores.items()
        if score >= _RELEVANCE_THRESHOLD
    ][: k * 2]
    if not candidates:
        return []

    ranked = rerank(query, candidates, k)
    final_ids = [h["component_uri"] for h in ranked]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, chunk_text, span_sha256, act_name, case_id,
                   chunk_type, section_number
            FROM chunks
            WHERE id::text = ANY(%(ids)s)
            """,
            {"ids": final_ids},
        )
        full_rows = {str(row[0]): row for row in cur.fetchall()}

    return [
        _hit(full_rows[h["component_uri"]], h["score"])
        for h in ranked
        if h["component_uri"] in full_rows
    ]
