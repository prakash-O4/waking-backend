from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, cast

from openai import AzureOpenAI
from psycopg2.extensions import connection

from app.config import azure_base_url, get_settings
from app.retrieval.eligibility_gate import eligible_chunk_ids
from app.retrieval.reranker import rerank
from app.utils.llm import llm_text
from app.utils.loggers import logger as logger

_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
_RELEVANCE_THRESHOLD = 0.005
_MAX_SECTION_RANGE = 50
_MAX_SECTION_NUMBERS = 10
_SECTION_RANGE_RE = re.compile(
    r"(?:दफा|धारा)\s*(\d+)\s*(?:देखि|-|–|—)\s*(\d+)\s*(?:सम्म)?"
)
_lf_client: Any = None


def _load_act_aliases() -> dict[str, str]:
    path = Path(__file__).with_name("act_aliases.json")
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return cast(dict[str, str], json.load(f))
    except json.JSONDecodeError:
        return {}


# Matching convenience only; never render aliases as citation/source metadata.
_ACT_ALIASES = _load_act_aliases()


def get_lf_client() -> Any | None:
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return None
    global _lf_client
    if _lf_client is None:
        try:
            from langfuse import Langfuse
        except ImportError:
            return None
        s = get_settings()
        _lf_client = Langfuse(
            public_key=s.LANGFUSE_PUBLIC_KEY,
            secret_key=s.LANGFUSE_SECRET_KEY,
            host=s.LANGFUSE_HOST,
        )
    return _lf_client


def _end_span(
    trace: Any,
    stage: str,
    *,
    input: Any = None,
    output: Any = None,
    **metadata: Any,
) -> None:
    """Create a span and immediately end it so Langfuse records endTime."""
    if trace is None:
        return
    try:
        kwargs: dict[str, Any] = {
            "name": f"stage.{stage}",
            "as_type": "span",
            "metadata": metadata,
        }
        if input is not None:
            kwargs["input"] = input
        span = trace.start_observation(**kwargs)
        if output is not None:
            span.update(output=output)
        span.end()
    except Exception:
        pass


def _preprocess(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_DIGIT_MAP)


def _is_devanagari(text: str) -> bool:
    count = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    return count / max(len(text), 1) > 0.5


def _needs_nepali_variant(query: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in query)


def translate_query(query: str) -> str | None:
    if not _needs_nepali_variant(query):
        return None
    s = get_settings()
    if not s.AZURE_OPENAI_LLM_KEY:
        return None
    try:
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint=azure_base_url(s.AZURE_OPENAI_LLM_ENDPOINT),
            azure_deployment=s.AZURE_OPENAI_LLM_DEPLOYMENT,
            api_key=s.AZURE_OPENAI_LLM_KEY,
            api_version=s.AZURE_OPENAI_API_VERSION,
            temperature=0.0,
            max_tokens=1000,
        )
        resp = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Translate the following legal query to formal Nepali in Devanagari script. "
                        "Output only the translated text. Do not add explanations."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        translated = llm_text(resp).strip()
        return translated if translated else None
    except Exception as e:
        logger.warning(f"translate_query failed: {e}")
        return None


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


def _parse_section_reference(query: str) -> str | None:
    nums = _parse_section_numbers(query)
    return nums[0] if nums else None


def _parse_section_numbers(query: str) -> list[str]:
    match = re.search(r"(?:दफा|धारा)\s*(\d+)((?:\s*(?:,|र|तथा)\s*\d+)*)", query)
    if not match:
        return []
    nums = [match.group(1), *re.findall(r"\d+", match.group(2))]
    return nums[:_MAX_SECTION_NUMBERS]


def _parse_section_range(query: str) -> tuple[int, int] | None:
    match = _SECTION_RANGE_RE.search(query)
    if not match:
        return None
    low, high = sorted((int(match.group(1)), int(match.group(2))))
    return (low, high) if high - low <= _MAX_SECTION_RANGE else None


def _parse_proviso_reference(query: str) -> bool:
    return bool(re.search(r"परन्तुक|स्पष्टीकरण", query))


def _parse_subsection_reference(query: str) -> str | None:
    match = re.search(r"उपदफा\s*\(?(\d+)\)?", query)
    return match.group(1) if match else None


def _parse_schedule_numbers(query: str) -> list[str]:
    return re.findall(r"अनुसूची\s*(\d+)", query)


def _parse_schedule_reference(query: str) -> str | None:
    nums = _parse_schedule_numbers(query)
    return nums[0] if nums else None


def _resolve_act_titles(conn: connection, query: str) -> list[str]:
    search_text = (
        query
        + " "
        + " ".join(
            canonical for alias, canonical in _ACT_ALIASES.items() if alias in query
        )
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, title_ne
            FROM work
            WHERE strpos(%(query)s, title_ne) > 0
               OR strpos(%(query)s, regexp_replace(title_ne, ',\\s*[०-९]+\\s*$', '')) > 0
            ORDER BY length(title_ne) DESC
            LIMIT 5
            """,
            {"query": search_text},
        )
        rows = cur.fetchall()
    return [str(row[0]) for row in rows]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _hit(
    row: tuple[Any, ...], score: float, vector_score: float = 0.0
) -> dict[str, Any]:
    chunk_id, text, text_hash, act_name, case_id, chunk_type, section_number = row[:7]
    source_id = row[7] if len(row) == 8 else ""
    return {
        "component_uri": str(chunk_id),
        "text_ne": text,
        "text_hash": text_hash,
        "score": float(score),
        "vector_score": float(vector_score),
        "work_title_ne": act_name or case_id or "",
        "chunk_type": chunk_type,
        "section_number": section_number or "",
        "document_source_id": str(source_id) if source_id else "",
    }


def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5, lf_trace: Any = None
) -> list[dict[str, Any]]:
    title_query = unicodedata.normalize("NFC", query)
    query = _preprocess(query)
    translated_raw = translate_query(query)
    title_query_ne = (
        unicodedata.normalize("NFC", translated_raw) if translated_raw else None
    )
    query_ne = _preprocess(translated_raw) if translated_raw else None
    query_variants = [query]
    if query_ne and query_ne != query:
        query_variants.append(query_ne)

    section_range = next(
        (match for v in query_variants if (match := _parse_section_range(v))), None
    )
    section_nums: list[str] = []
    if section_range or any(_SECTION_RANGE_RE.search(v) for v in query_variants):
        section_nums = []
    else:
        for v in query_variants:
            for num in _parse_section_numbers(v):
                if num not in section_nums:
                    section_nums.append(num)
        section_nums = section_nums[:_MAX_SECTION_NUMBERS]
    section_num = section_nums[0] if len(section_nums) == 1 else None
    schedule_nums: list[str] = []
    for v in query_variants:
        for num in _parse_schedule_numbers(v):
            if num not in schedule_nums:
                schedule_nums.append(num)
    proviso_ref = any(_parse_proviso_reference(v) for v in query_variants)
    act_work_ids: list[str] = []
    retrieval_span = None
    if lf_trace is not None:
        try:
            retrieval_span = lf_trace.start_observation(
                name="retrieval",
                as_type="span",
                input={
                    "query": query,
                    "query_ne": query_ne,
                    "as_of": str(as_of),
                    "k": k,
                },
                metadata={
                    "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
                    "as_of": str(as_of),
                    "k": k,
                },
            )
        except Exception:
            pass

    t0 = time.monotonic()
    eligible = list(eligible_chunk_ids(conn, as_of))
    _end_span(
        retrieval_span,
        "eligibility_gate",
        eligible_count=len(eligible),
        translation_ran=query_ne is not None,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    if not eligible:
        if retrieval_span:
            try:
                retrieval_span.update(output=[])
                retrieval_span.end()
            except Exception:
                pass
        return []

    title_variants = [title_query]
    if title_query_ne:
        title_variants.append(title_query_ne)
    for title_variant in title_variants:
        for work_id in _resolve_act_titles(conn, title_variant):
            if work_id not in act_work_ids:
                act_work_ids.append(work_id)
    act_work_ids = act_work_ids[:5]
    qvec = _embed_query(query)
    qvec_ne = _embed_query(query_ne) if query_ne and query_ne != query else None
    limit = k * 3
    with conn.cursor() as cur:

        def vector_search(vec: list[float]) -> list[tuple[Any, ...]]:
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
                {"qvec": vec, "eligible": eligible, "limit": limit},
            )
            return cur.fetchall()

        def lexical_search(text: str) -> list[tuple[Any, ...]]:
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
                {"query": text, "eligible": eligible, "limit": limit},
            )
            return cur.fetchall()

        def exact_lookup_search() -> list[tuple[Any, ...]]:
            filters = ["c.id::text = ANY(%(eligible)s)"]
            params: dict[str, Any] = {"eligible": eligible, "limit": limit}
            if act_work_ids:
                # ponytail: section refs apply globally across matched Acts; add per-Act
                # pairing only when query grammar needs it.
                filters.append("c.work_id = ANY(%(work_ids)s)")
                params["work_ids"] = act_work_ids
            if section_range is not None:
                filters.append(
                    "(NULLIF(regexp_replace(c.section_number, '\\D', '', 'g'), '')::int "
                    "BETWEEN %(low)s AND %(high)s OR "
                    "NULLIF(regexp_replace(c.parent_section, '\\D', '', 'g'), '')::int "
                    "BETWEEN %(low)s AND %(high)s)"
                )
                params["low"], params["high"] = section_range
            elif section_num is not None:
                filters.append(
                    "(c.section_number = %(num)s OR c.parent_section = %(num)s)"
                )
                params["num"] = section_num
            elif section_nums:
                filters.append(
                    "(c.section_number = ANY(%(nums)s) OR c.parent_section = ANY(%(nums)s))"
                )
                params["nums"] = section_nums
            if proviso_ref and (section_range is not None or section_nums):
                filters.append("c.level = 'proviso'")
            if schedule_nums:
                filters.append(
                    "EXISTS ("
                    "SELECT 1 FROM unnest(%(schedule_nums)s) AS sn "
                    "WHERE strpos(c.chunk_text, 'अनुसूची ' || sn) > 0"
                    ")"
                )
                params["schedule_nums"] = schedule_nums
            cur.execute(
                f"""
                SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
                       c.chunk_type, c.section_number, d.source_id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {" AND ".join(filters)}
                LIMIT %(limit)s
                """,
                params,
            )
            return cur.fetchall()

        t0 = time.monotonic()
        vector_rows = vector_search(qvec)
        vector_rows_ne = vector_search(qvec_ne) if qvec_ne is not None else []
        _end_span(
            retrieval_span,
            "vector_search",
            output={
                "query": [
                    {
                        "chunk_id": row[0],
                        "section_number": row[6],
                        "score": round(float(row[-1]), 4),
                    }
                    for row in vector_rows[:3]
                ],
                "query_ne": [
                    {
                        "chunk_id": row[0],
                        "section_number": row[6],
                        "score": round(float(row[-1]), 4),
                    }
                    for row in vector_rows_ne[:3]
                ],
            },
            candidate_count=len(vector_rows) + len(vector_rows_ne),
            top_score=float(vector_rows[0][8 if len(vector_rows[0]) > 8 else 7])
            if vector_rows
            else 0.0,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        lexical_rows: list[tuple[Any, ...]] = []
        lexical_rows_ne: list[tuple[Any, ...]] = []
        lexical_ran = any(ch.isalpha() for ch in query)
        t0 = time.monotonic()
        if lexical_ran:
            lexical_rows = lexical_search(query)
        if query_ne is not None and query_ne != query:
            lexical_rows_ne = lexical_search(query_ne)
        _end_span(
            retrieval_span,
            "lexical_search",
            output={
                "query": [
                    {
                        "chunk_id": row[0],
                        "section_number": row[6],
                        "score": round(float(row[-1]), 4),
                    }
                    for row in lexical_rows[:3]
                ],
                "query_ne": [
                    {
                        "chunk_id": row[0],
                        "section_number": row[6],
                        "score": round(float(row[-1]), 4),
                    }
                    for row in lexical_rows_ne[:3]
                ],
            },
            ran=lexical_ran or query_ne is not None,
            candidate_count=len(lexical_rows) + len(lexical_rows_ne),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        exact_ran = bool(section_range or section_nums or schedule_nums or act_work_ids)
        exact_rows: list[tuple[Any, ...]] = []
        t0 = time.monotonic()
        if exact_ran:
            exact_rows = exact_lookup_search()
        _end_span(
            retrieval_span,
            "exact_lookup",
            input={
                "section_range": section_range,
                "section_num": section_num,
                "section_nums": section_nums,
                "schedule_nums": schedule_nums,
                "act_work_ids": act_work_ids,
                "proviso_ref": proviso_ref,
            },
            output=[
                {"chunk_id": row[0], "section_number": row[6]} for row in exact_rows[:3]
            ],
            ran=exact_ran,
            candidate_count=len(exact_rows),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    rows = {
        str(row[0]): row
        for row in [
            *vector_rows,
            *lexical_rows,
            *vector_rows_ne,
            *lexical_rows_ne,
            *exact_rows,
        ]
    }
    vector_scores = {
        str(row[0]): float(row[-1]) for row in [*vector_rows, *vector_rows_ne]
    }
    ranked_lists = [[str(r[0]) for r in vector_rows], [str(r[0]) for r in lexical_rows]]
    if qvec_ne is not None:
        ranked_lists.append([str(r[0]) for r in vector_rows_ne])
    if query_ne is not None and query_ne != query:
        ranked_lists.append([str(r[0]) for r in lexical_rows_ne])
    if exact_ran:
        ranked_lists.append([str(r[0]) for r in exact_rows])
    t0 = time.monotonic()
    rrf_scores = _rrf(ranked_lists)
    _end_span(
        retrieval_span,
        "rrf_fusion",
        output=[
            {"chunk_id": chunk_id, "rrf_score": round(score, 4)}
            for chunk_id, score in rrf_scores[:5]
        ],
        merged_count=len(rrf_scores),
        top_rrf_score=rrf_scores[0][1] if rrf_scores else 0.0,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )

    t0 = time.monotonic()
    candidates = [
        _hit(rows[chunk_id], score, vector_scores.get(chunk_id, 0.0))
        for chunk_id, score in rrf_scores
        if score >= _RELEVANCE_THRESHOLD
    ][: k * 2]
    _end_span(
        retrieval_span,
        "relevance_gate",
        output=[
            {
                "chunk_id": c["component_uri"],
                "score": round(c.get("score", 0.0), 4),
            }
            for c in candidates[:3]
        ],
        passed_count=len(candidates),
        abstained=not candidates,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    if not candidates:
        if retrieval_span:
            try:
                retrieval_span.update(output=[])
                retrieval_span.end()
            except Exception:
                pass
        return []

    t0 = time.monotonic()
    ranked = rerank(query, candidates, k)
    reranker_tier = ranked[0].get("reranker_tier", "none") if ranked else "none"
    _end_span(
        retrieval_span,
        "rerank",
        output=[
            {
                "chunk_id": r["component_uri"],
                "score": round(r.get("score", 0.0), 4),
            }
            for r in ranked[:3]
        ],
        tier=reranker_tier,
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

    result_hits = [
        {
            **_hit(
                full_rows[h["component_uri"]], h["score"], h.get("vector_score", 0.0)
            ),
            "reranker_tier": h.get("reranker_tier", "unknown"),
        }
        for h in ranked
        if h["component_uri"] in full_rows
    ]
    top_vec = max((h.get("vector_score", 0.0) for h in result_hits), default=0.0)
    if retrieval_span is not None:
        try:
            retrieval_span.update(
                output=[
                    {
                        "work_title_ne": h.get("work_title_ne", ""),
                        "section_number": h.get("section_number", ""),
                        "score": round(h.get("score", 0.0), 4),
                        "text_snippet": str(h.get("text_ne", ""))[:160],
                    }
                    for h in result_hits
                ],
                metadata={
                    "eligible_count": len(eligible),
                    "final_count": len(result_hits),
                    "top_vector_score": round(top_vec, 4),
                },
            )
            retrieval_span.end()
        except Exception:
            pass
    return result_hits
