"""Deterministic whole-act repeal extractor.

Finds only clean clauses of the form "<Act>, <year> खारेज गरिएको छ".
Partial दफा repeals and savings clauses are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.authority.writer import propose_lifecycle_repeal
from app.ingestion.enabling_extractor import _normalize_title, _resolve_work

_REPEAL_RE = re.compile(
    r"(?:\*\*[^\n।]{0,100}?\*\*\s*)?"
    r"(?:\([०-९]+\)\s*)?"
    r"(?P<title>[^\n।]{2,140}?(?:ऐन|अध्यादेश|नियमावली|नियम),\s*[०-९]{4})"
    r"\s+खारेज\s+गरिएको\s+छ"
)


@dataclass(frozen=True)
class RepealMatch:
    repealed_work_id: str | None
    repealed_title_ne: str | None
    raw_clause_text: str
    resolution_status: str


def _clean_title(raw: str) -> str:
    if ":" in raw and "खारेजी" in raw.split(":", 1)[0]:
        raw = raw.split(":", 1)[1]
    raw = re.sub(r"^\s*\([०-९]+\)\s*", "", raw)
    return _normalize_title(raw)


def classify_repeal(content: str, conn: Any) -> RepealMatch:
    match = _REPEAL_RE.search(content)
    if not match:
        return RepealMatch(None, None, "", "no_repeal_clause")

    title = _clean_title(match.group("title"))
    work_id = _resolve_work(conn, title)
    status = "auto_extracted" if work_id else "repealed_work_not_in_corpus"
    return RepealMatch(work_id, title, match.group(0), status)


def extract_repeal_proposals(
    *, law: Any, content: str, source_pub_id: str, conn: Any
) -> RepealMatch:
    match = classify_repeal(content, conn)
    if match.repealed_work_id:
        propose_lifecycle_repeal(
            conn,
            match.repealed_work_id,
            source_pub_id,
            repealing_work_uri=law.uri,
            raw_clause_text=match.raw_clause_text,
        )
    return match
