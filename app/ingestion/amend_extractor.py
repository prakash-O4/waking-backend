"""Deterministic <amend> tag lifecycle proposal extractor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from app.authority.bs_ad_calendar import BeyondCalendarRange, lookup
from app.authority.parser import (
    _HEADER_RE,
    _clean_text,
    _component,
    _component_kind,
    _disambiguate_component_uris,
)
from app.authority.writer import propose_lifecycle_amend
from app.ingestion.commencement_extractor import ORDINAL_DAYS
from app.ingestion.enabling_extractor import _normalize_title

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_AMEND_RE = re.compile(r"<amend>(?P<text>.*?)</amend>", re.DOTALL)
_TABLE_HEADING_RE = re.compile(r"(?m)^\s*(?:संशोधन\s+गर्ने\s+(?:ऐन|नियम)|संशोधन)\s*$")
_ROW_START_RE = re.compile(r"^\s*(?P<pos>[०-९0-9]+)[.)।]\s*(?P<rest>.*)")
_ROW_BODY_RE = re.compile(
    r"(?P<name>.{2,240}?,?\s*[०-९]{4})(?:\s+(?P<date>[०-९]{4}[।./-][०-९]{1,2}[।./-][०-९]{1,2}))?"
)
_DATE_IN_ROW_RE = re.compile(r"[०-९]{4}[।./-][०-९]{1,2}[।./-][०-९]{1,2}")
_NAMED_RE = re.compile(
    r"^\s*(?P<name>[^।\n]{2,160}?(?:ऐन|नियमावली|नियम)),?\s*(?P<year>[०-९]{4})\s+द्वारा\s+(?:संशोधित|थप)"
)
_ORDINAL_RE = re.compile(r"^\s*(?P<ordinal>[^\s।]+)\s+संशोधनद्वारा\s+(?:संशोधित|थप)")
_DATE_RE = re.compile(r"^\s*[०-९]{4}[।./-][०-९]{1,2}[।./-][०-९]{1,2}\s*[।.]?\s*$")


@dataclass(frozen=True)
class AmendmentTableEntry:
    position: int
    normalized_name: str
    bs_date: str | None
    effective_date: date | None


@dataclass(frozen=True)
class AmendProposal:
    component_uri: str
    method: str
    amending_title_ne: str
    effective_date: date | None
    raw_clause_text: str


@dataclass(frozen=True)
class AmendExtractionResult:
    proposals: list[AmendProposal]
    skipped: dict[str, int]


def _to_int(text: str) -> int:
    return int(text.translate(_DEVA_DIGITS))


def _bs_to_ad(text: str) -> date | None:
    parts = re.split(r"[।./-]", text)
    try:
        ad_date, _ = lookup(*(_to_int(part) for part in parts))
    except (BeyondCalendarRange, ValueError):
        return None
    return cast(date, ad_date)


def _parse_table_row(
    position: int, text: str, *, require_date: bool = False
) -> AmendmentTableEntry | None:
    text = " ".join(text.split())
    if require_date and not _DATE_IN_ROW_RE.search(text):
        return None
    match = _ROW_BODY_RE.search(text)
    if not match:
        return None
    bs_date = match.group("date")
    return AmendmentTableEntry(
        position=position,
        normalized_name=_normalize_title(match.group("name")),
        bs_date=bs_date,
        effective_date=_bs_to_ad(bs_date) if bs_date else None,
    )


def parse_amendment_table(content: str) -> list[AmendmentTableEntry]:
    heading = _TABLE_HEADING_RE.search(content[:5000])
    if not heading:
        return []
    first_header = _HEADER_RE.search(content, heading.end())
    table_text = content[heading.end() : first_header.start() if first_header else 5000]
    rows: list[AmendmentTableEntry] = []
    position: int | None = None
    body = ""
    for line in table_text.splitlines():
        start = _ROW_START_RE.match(line)
        if start and len(start.group("pos").translate(_DEVA_DIGITS)) <= 2:
            parsed = _parse_table_row(position, body) if position is not None else None
            if parsed:
                rows.append(parsed)
            position = _to_int(start.group("pos"))
            body = start.group("rest")
        elif position is not None:
            body += " " + line.strip()
        parsed = (
            _parse_table_row(position, body, require_date=True)
            if position is not None
            else None
        )
        if parsed:
            rows.append(parsed)
            position = None
            body = ""
    parsed = _parse_table_row(position, body) if position is not None else None
    if parsed:
        rows.append(parsed)
    return rows


def _component_spans(content: str, law: Any) -> list[tuple[int, int, str]]:
    matches = list(_HEADER_RE.finditer(content))
    raw_components = []
    spans: list[tuple[int, int]] = []
    if matches:
        preamble = _component(law.uri, "full", "0", content[: matches[0].start()])
        if preamble:
            raw_components.append(preamble)
            spans.append((0, matches[0].start()))
        schedule_number = None
        for idx, match in enumerate(matches):
            kind, number, schedule_number = _component_kind(match, schedule_number)
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            item = _component(law.uri, kind, number, content[match.start() : end])
            if item:
                raw_components.append(item)
                spans.append((match.start(), end))
    else:
        item = _component(law.uri, "full", "1", content)
        if item:
            raw_components.append(item)
            spans.append((0, len(content)))
    _disambiguate_component_uris(raw_components)
    return [
        (start, end, component.uri)
        for (start, end), component in zip(spans, raw_components)
    ]


def _enclosing_uri(spans: list[tuple[int, int, str]], offset: int) -> str | None:
    for start, end, uri in spans:
        if start <= offset < end:
            return uri
    return None


def _normalize_ordinal(text: str) -> str:
    text = text.replace("ँ", "ं")
    for old, new in {
        "प्रथम": "पहिलो",
        "पहिले": "पहिलो",
        "दोश्रो": "दोस्रो",
        "तेसो": "तेस्रो",
        "चौथौं": "चौथो",
        "चौथौ": "चौथो",
        "चौथों": "चौथो",
        "छैठ": "छैट",
        "नौव": "नव",
        "नह": "नव",
        "नब": "नव",
        "छब्बिस": "छब्बीस",
        "छबिस": "छब्बीस",
        "चौबिस": "चौबीस",
        "पच्चिस": "पच्चीस",
        "सात्त": "सात",
        "पाच": "पांच",
    }.items():
        text = text.replace(old, new)
    if text.endswith("ौ"):
        text += "ं"
    return text


_NORMALIZED_ORDINAL_DAYS = {
    _normalize_ordinal(word): value for word, value in ORDINAL_DAYS.items()
}


def classify_amend_text(
    text: str, table: list[AmendmentTableEntry]
) -> tuple[str, AmendmentTableEntry | None]:
    match = _NAMED_RE.search(text)
    if match:
        wanted = _normalize_title(f"{match.group('name')}, {match.group('year')}")
        return "named-act", next(
            (row for row in table if row.normalized_name == wanted), None
        )

    match = _ORDINAL_RE.search(text)
    if match:
        ordinal = _NORMALIZED_ORDINAL_DAYS.get(
            _normalize_ordinal(match.group("ordinal"))
        )
        if ordinal is None:
            return "unknown-ordinal", None
        return "ordinal", next((row for row in table if row.position == ordinal), None)

    if _DATE_RE.match(text):
        return "gazette-date-only", None
    return "other", None


def extract_amend_proposals(
    *, law: Any, content: str, source_pub_id: str, conn: Any
) -> AmendExtractionResult:
    table = parse_amendment_table(content)
    spans = _component_spans(content, law)
    skipped: dict[str, int] = {}
    proposals: list[AmendProposal] = []
    seen: set[tuple[str, int, str]] = set()

    for tag in _AMEND_RE.finditer(content):
        raw_text = tag.group("text").strip()
        if not table:
            skipped["no_table"] = skipped.get("no_table", 0) + 1
            continue
        method, entry = classify_amend_text(raw_text, table)
        if entry is None:
            skipped[f"unresolved_{method}"] = skipped.get(f"unresolved_{method}", 0) + 1
            continue
        component_uri = _enclosing_uri(spans, tag.start())
        if component_uri is None:
            skipped["no_component"] = skipped.get("no_component", 0) + 1
            continue
        key = (component_uri, entry.position, raw_text)
        if key in seen:
            skipped["duplicate"] = skipped.get("duplicate", 0) + 1
            continue
        seen.add(key)
        proposal = AmendProposal(
            component_uri=component_uri,
            method=method,
            amending_title_ne=entry.normalized_name,
            effective_date=entry.effective_date,
            raw_clause_text=_clean_text(raw_text),
        )
        proposals.append(proposal)
        propose_lifecycle_amend(
            conn,
            component_uri,
            source_pub_id,
            effective_date=entry.effective_date,
            amendment_dependency=None
            if entry.effective_date
            else f"amendment_date_unresolved:{entry.bs_date or entry.normalized_name}",
            raw_clause_text=proposal.raw_clause_text,
        )

    return AmendExtractionResult(proposals, skipped)
