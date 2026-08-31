from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import unquote

from app.authority.bs_ad_calendar import BeyondCalendarRange, lookup
from app.authority.models import WorkType


@dataclass
class ParsedLaw:
    uri: str
    work_type: WorkType
    title_ne: str
    title_en: str | None
    enactment_ad: date | None
    source_sha256: str
    components: list[ParsedComponent]


@dataclass
class ParsedComponent:
    uri: str
    component_type: str
    number: str | None
    text_ne: str
    text_hash: str


_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DATE_RE = re.compile(
    r"([१२३४५६७८९०]{4})[।./-]([१२३४५६७८९०]{1,2})[।./-]([१२३४५६७८९०]{1,2})"
)
_AMEND_RE = re.compile(r"</?amend>")
_HEADER_RE = re.compile(
    r"(?m)(\*\*\s*अनुसूची[-–]\s*([०-९0-9]+)[^\n*]*\*\*|\*\*\s*(?:दफा\s*)?([०-९0-9]+(?:\.[०-९0-9]+)?)[\.।:][^\n*]*\*\*|^दफा\s+([०-९0-9]+(?:\.[०-९0-9]+)?)[\.।]|^परिच्छेद[-–]\s*([०-९0-9]+)\s*$|^धारा\s+([०-९0-9]+)[\.।])"
)


def _ascii_digits(text: str) -> str:
    return text.translate(_DIGITS)


def make_uri(name: str, doc_type: str, record_id: str | None = None) -> str:
    parts = name.rsplit("_", 1)
    bs_year = (
        _ascii_digits(parts[-1])
        if len(parts) == 2 and re.fullmatch(r"[०-९]{4}", parts[-1])
        else "unknown"
    )
    slug = (
        record_id
        or re.sub(r"[^a-z0-9]+", "-", _ascii_digits(parts[0]).lower()).strip("-")
        or "law"
    )
    return f"/np/{doc_type}/{bs_year}/{slug}"


def _work_type(doc_type: str) -> WorkType:
    value = doc_type.lower()
    if value == "act":
        return WorkType.ACT
    if value in {"rule", "regulation", "regulations"}:
        return WorkType.RULE
    if value == "constitution":
        return WorkType.CONSTITUTION
    if value in {"directive", "notification"}:
        return WorkType(value.capitalize())
    return WorkType.ACT


def _enactment_ad(content: str) -> date | None:
    head = content[:800]
    marker = re.search(r"प्रमाणीकरण\s+र\s+प्रकाश(?:न|ित)\s+मिति", head)
    match = (
        _DATE_RE.search(head, marker.end() if marker else 0)
        if marker
        else _DATE_RE.search(head)
    )
    if not match:
        return None
    year, month, day = (int(_ascii_digits(group)) for group in match.groups())
    try:
        ad_date, _ = lookup(year, month, day)
    except BeyondCalendarRange:
        return None
    return ad_date


def _clean_text(text: str) -> str:
    return _AMEND_RE.sub("", text).strip()


def _component_kind(
    match: re.Match[str], schedule_number: str | None
) -> tuple[str, str | None, str | None]:
    if match.group(2):
        number = _ascii_digits(match.group(2))
        return "anushuchi", number, number
    if match.group(3):
        number = _ascii_digits(match.group(3))
        if schedule_number:
            return "anushuchi", f"{schedule_number}.{number}", schedule_number
        return "dafa", number, schedule_number
    if match.group(4):
        return "dafa", _ascii_digits(match.group(4)), schedule_number
    if match.group(5):
        return "parichheda", _ascii_digits(match.group(5)), schedule_number
    if match.group(6):
        return "dhara", _ascii_digits(match.group(6)), schedule_number
    return "dafa", None, schedule_number


def _component(
    uri: str, kind: str, number: str | None, text: str
) -> ParsedComponent | None:
    text_ne = _clean_text(text)
    if len(text_ne) < 20:
        return None
    return ParsedComponent(
        uri=f"{uri}/{kind}/{number or '1'}",
        component_type=kind,
        number=number,
        text_ne=text_ne,
        text_hash=hashlib.sha256(text_ne.encode("utf-8")).hexdigest(),
    )


def _disambiguate_component_uris(components: list[ParsedComponent]) -> None:
    seen: Counter[str] = Counter()
    for component in components:
        seen[component.uri] += 1
        if seen[component.uri] > 1:
            # Source texts sometimes repeat a दफा number for different provisions;
            # keep both addressable without guessing which number is wrong.
            component.uri = f"{component.uri}/occurrence/{seen[component.uri]}"


def parse_law(record: dict[str, Any]) -> ParsedLaw:
    content = str(record["content"])
    uri = make_uri(
        str(record["name"]),
        str(record.get("document_type") or "act"),
        str(record.get("_id") or ""),
    )
    matches = list(_HEADER_RE.finditer(content))
    components: list[ParsedComponent] = []
    if matches:
        preamble = _component(uri, "full", "0", content[: matches[0].start()])
        if preamble:
            components.append(preamble)
        schedule_number: str | None = None
        for idx, match in enumerate(matches):
            kind, number, schedule_number = _component_kind(match, schedule_number)
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            item = _component(uri, kind, number, content[match.start() : end])
            if item:
                components.append(item)
    else:
        item = _component(uri, "full", "1", content)
        if item:
            components.append(item)
    _disambiguate_component_uris(components)

    return ParsedLaw(
        uri=uri,
        work_type=_work_type(str(record.get("document_type") or "act")),
        title_ne=unquote(str(record["name"])).replace("_", " "),
        title_en=record.get("english_name"),
        enactment_ad=_enactment_ad(content),
        source_sha256=hashlib.sha256(
            unicodedata.normalize("NFC", content).encode("utf-8")
        ).hexdigest(),
        components=components,
    )


def parse_jsonl_line(line: str) -> ParsedLaw:
    return parse_law(json.loads(line))
