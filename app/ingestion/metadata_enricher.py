"""
Metadata enrichment via provider-agnostic LLM (design §3).

Replaces the old Anthropic-specific enricher. All LLM outputs are
proposals: JSON parse failures log a warning and leave columns empty (NULL) —
the pipeline never crashes and never guesses metadata.

Batching note (design §3.3): keywords + relevant_questions for all chunks of
one document go in a single LLM call. Per-document functions use individual
invoke calls; a batch driver that accumulates ≥10 documents is a follow-up for
the full-corpus run.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import get_settings
from app.ingestion.laws_chunker import LawChunk
from app.ingestion.nkp_chunker import NKPChunk
from app.utils.loggers import logger

MAX_RETRIES = 3
BATCH_API_MIN_DOCUMENTS = 10  # see module docstring

_llm: Any = None


def _get_llm() -> Any:
    """Lazy LangChain LLM — import and key resolution happen at first call,
    so the module is importable without provider SDKs or API keys (tests)."""
    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model

        settings = get_settings()
        _llm = init_chat_model(settings.LLM_MODEL, temperature=0)
    return _llm


def _parse_json(raw: str, expected: type) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    start_chars = {"list": "[", "dict": "{"}
    end_chars = {"list": "]", "dict": "}"}
    kind = "list" if expected is list else "dict"
    start, end = raw.find(start_chars[kind]), raw.rfind(end_chars[kind])
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON {kind} found in response")
    return json.loads(raw[start : end + 1])


def _call_llm(prompt: str) -> str:
    """One LLM call with exponential backoff on transient failures (max 3 retries)."""
    from langchain_core.messages import HumanMessage

    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _get_llm().invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as exc:  # noqa: BLE001 — provider-agnostic retry
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                f"llm call failed ({exc}), retry {attempt + 1} in {delay:.0f}s"
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _chunk_metadata_prompt(chunks: list[Any], intro: str) -> str:
    numbered = "\n\n".join(
        f"[{chunk.chunk_index}]\n{chunk.chunk_text[:2000]}" for chunk in chunks
    )
    return (
        f"{intro}\n\n"
        "For EACH numbered chunk below, return:\n"
        '- "keywords": 3–8 Nepali legal keywords\n'
        '- "relevant_questions": 3–5 Nepali questions this chunk answers\n'
        "Return ONLY a JSON array aligned by chunk index: "
        '[{"chunk_index": 0, "keywords": [...], "relevant_questions": [...]}]'
        f"\n\nChunks:\n{numbered}"
    )


def _empty_chunk_metadata(chunks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": chunk.chunk_index,
            "keywords": None,
            "relevant_questions": None,
        }
        for chunk in chunks
    ]


def _apply_chunk_metadata(
    chunks: list[Any], raw: str, extra: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    metadata = _empty_chunk_metadata(chunks)
    try:
        parsed = _parse_json(raw, list)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"llm chunk metadata not parseable, columns left NULL: {exc}")
        return metadata
    by_index = {
        int(item["chunk_index"]): item
        for item in parsed
        if isinstance(item, dict) and "chunk_index" in item
    }
    for entry in metadata:
        item = by_index.get(entry["chunk_index"])
        if not item:
            continue
        entry["keywords"] = item.get("keywords") or None
        entry["relevant_questions"] = item.get("relevant_questions") or None
    if extra:
        for entry in metadata:
            entry.update(extra)
    return metadata


def enrich_law_chunks(
    act_record: dict[str, Any], chunks: list[LawChunk]
) -> list[dict[str, Any]]:
    """
    Two LLM calls per act:
    1. Document-level summary of the act.
    2. Per-chunk keywords + relevant_questions.
    Returns metadata dicts aligned by chunk_index; each carries the act summary.
    """
    if not chunks:
        return []
    act_name = act_record.get("name", "")
    sample_text = "\n\n".join(c.chunk_text for c in chunks[:5])

    doc_extra: dict[str, Any] = {"summary": None}
    try:
        raw = _call_llm(
            f"You are indexing the Nepali act «{act_name}» for legal search.\n"
            "Write a 2-3 sentence Nepali summary: what this act governs and its key provisions.\n"
            'Return ONLY JSON: {"summary": "..."}'
            f"\n\nAct excerpt:\n{sample_text[:8000]}"
        )
        parsed = _parse_json(raw, dict)
        doc_extra["summary"] = parsed.get("summary") or None
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"llm act summary not parseable, left NULL: {exc}")

    prompt = _chunk_metadata_prompt(
        chunks,
        f"You are indexing the Nepali act «{act_name}» for legal search.",
    )
    raw = _call_llm(prompt)
    return _apply_chunk_metadata(chunks, raw, extra=doc_extra)


def enrich_nkp_chunks(
    case_record: dict[str, Any], chunks: list[NKPChunk]
) -> list[dict[str, Any]]:
    """
    Two LLM calls per case:
    1. cited_statutes + headnotes cleanup over the full redacted text.
    2. keywords + relevant_questions per chunk (single batched call).
    Returns metadata dicts aligned by chunk_index; each also carries the
    document-level cited_statutes / headnotes.
    """
    full_text = "\n\n".join(chunk.chunk_text for chunk in chunks)

    doc_extra: dict[str, Any] = {"cited_statutes": None, "headnotes": None, "summary": None}
    try:
        raw = _call_llm(
            "You are indexing a Nepali Supreme Court decision for legal search.\n"
            "From the redacted full text below, extract:\n"
            '- "cited_statutes": names of Nepali acts/regulations cited (Nepali)\n'
            '- "headnotes": cleaned-up सिद्धान्त statements as one text block\n'
            '- "summary": 2-3 sentence Nepali summary — case topic, legal question decided, and outcome\n'
            "Return ONLY JSON: "
            '{"cited_statutes": [...], "headnotes": "...", "summary": "..."}'
            f"\n\nText:\n{full_text[:30000]}"
        )
        parsed = _parse_json(raw, dict)
        doc_extra["cited_statutes"] = parsed.get("cited_statutes") or None
        doc_extra["headnotes"] = parsed.get("headnotes") or None
        doc_extra["summary"] = parsed.get("summary") or None
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"llm case-level metadata not parseable, left NULL: {exc}")

    metadata = _empty_chunk_metadata(chunks)
    if chunks:
        raw = _call_llm(
            _chunk_metadata_prompt(
                chunks,
                "You are indexing a Nepali Supreme Court decision for legal search.",
            )
        )
        metadata = _apply_chunk_metadata(chunks, raw)
    for entry in metadata:
        entry.update(doc_extra)
    return metadata
