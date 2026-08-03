"""
Metadata enrichment via Anthropic haiku (design §3).

Replaces the old langchain_openai/ChatOpenAI enricher. All LLM outputs are
proposals: JSON parse failures log a warning and leave columns empty (NULL) —
the pipeline never crashes and never guesses metadata.

Model: claude-haiku-4-5-20251001 for all extraction.
Batching note (design §3.3): keywords + relevant_questions for all chunks of
one document go in a single haiku call. The Anthropic Message Batches API
(50% cost) only pays off across ≥10 documents; these per-document functions
never see that many, so they use individual messages.create calls. A batch
driver that accumulates ≥10 documents and calls _client.messages.batches.create
is a follow-up for the full-corpus run.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.ingestion.laws_chunker import LawChunk
from app.ingestion.nkp_chunker import NKPChunk
from app.utils.loggers import logger

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
BATCH_API_MIN_DOCUMENTS = 10  # see module docstring

_client: Any = None


def _get_client() -> Any:
    """Lazy anthropic client — import and key resolution happen at first call,
    so the module is importable without the SDK or ANTHROPIC_API_KEY (tests)."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def _parse_json(raw: str, expected: type) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    start_chars = {"list": "[", "dict": "{"}
    end_chars = {"list": "]", "dict": "}"}
    kind = "list" if expected is list else "dict"
    start, end = raw.find(start_chars[kind]), raw.rfind(end_chars[kind])
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON {kind} found in response")
    return json.loads(raw[start : end + 1])


def _call_haiku(prompt: str, max_tokens: int = 4096) -> str:
    """One haiku call with exponential backoff on 429/5xx (max 3 retries)."""
    import anthropic

    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _get_client().messages.create(
                model=HAIKU_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text"
            )
        except anthropic.APIStatusError as exc:
            retriable = exc.status_code == 429 or exc.status_code >= 500
            if not retriable or attempt == MAX_RETRIES:
                raise
            logger.warning(
                f"haiku call failed ({exc.status_code}), retry {attempt + 1} "
                f"in {delay:.0f}s"
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
        logger.warning(f"haiku chunk metadata not parseable, columns left NULL: {exc}")
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
    One haiku call per act: send the numbered chunk list, get back keywords +
    relevant_questions per chunk. Returns metadata dicts aligned by chunk_index.
    """
    if not chunks:
        return []
    act_name = act_record.get("name", "")
    prompt = _chunk_metadata_prompt(
        chunks,
        f"You are indexing the Nepali act «{act_name}» for legal search.",
    )
    raw = _call_haiku(prompt)
    return _apply_chunk_metadata(chunks, raw)


def enrich_nkp_chunks(
    case_record: dict[str, Any], chunks: list[NKPChunk]
) -> list[dict[str, Any]]:
    """
    Two haiku calls per case:
    1. cited_statutes + headnotes cleanup over the full redacted text.
    2. keywords + relevant_questions per chunk (single batched call).
    Returns metadata dicts aligned by chunk_index; each also carries the
    document-level cited_statutes / headnotes.
    """
    full_text = "\n\n".join(chunk.chunk_text for chunk in chunks)

    doc_extra: dict[str, Any] = {"cited_statutes": None, "headnotes": None}
    try:
        raw = _call_haiku(
            "You are indexing a Nepali Supreme Court decision for legal search.\n"
            "From the redacted full text below, extract:\n"
            '- "cited_statutes": names of Nepali acts/regulations cited (Nepali)\n'
            '- "headnotes": cleaned-up सिद्धान्त statements as one text block\n'
            "Return ONLY JSON: "
            '{"cited_statutes": [...], "headnotes": "..."}'
            f"\n\nText:\n{full_text[:30000]}"
        )
        parsed = _parse_json(raw, dict)
        doc_extra["cited_statutes"] = parsed.get("cited_statutes") or None
        doc_extra["headnotes"] = parsed.get("headnotes") or None
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"haiku case-level metadata not parseable, left NULL: {exc}")

    metadata = _empty_chunk_metadata(chunks)
    if chunks:
        raw = _call_haiku(
            _chunk_metadata_prompt(
                chunks,
                "You are indexing a Nepali Supreme Court decision for legal search.",
            )
        )
        metadata = _apply_chunk_metadata(chunks, raw)
    for entry in metadata:
        entry.update(doc_extra)
    return metadata
