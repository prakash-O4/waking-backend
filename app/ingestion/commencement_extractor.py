"""Deterministic commencement-clause proposal extractor.

Every match is only a pending lifecycle proposal. Unknown/no-match cases emit a
sentinel proposal instead of disappearing silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from app.authority.writer import propose_lifecycle_commence

_WINDOW = 2500

_IMMEDIATE_RE = re.compile(r"यो\s+[^।]{0,40}?तुरुन्त\s+प्रारम्भ\s+हुनेछ")
_RELATIVE_RE = re.compile(
    r"यो\s+[^।]{0,40}?प्रमाणीकरण\s+भएको\s+([^\s।]{1,30})\s+दिनदेखि\s+प्रारम्भ\s+हुनेछ"
)
_GAZETTE_RE = re.compile(
    r"यो\s+[^।]{0,80}?नेपाल\s+सरकारले\s+नेपाल\s+राजपत्रमा\s+सूचना\s+"
    r"(?:प्रकाशन|प्रकाशित)\s+गरी\s+(?:तोकेको|तोकिदिएको|तोकिएको)\s+"
    r"मिति(?:देखि|मा)\s+प्रारम्भ\s+हुनेछ"
)
_PUBLICATION_RE = re.compile(
    r"यो\s+[^।]{0,40}?नेपाल\s+राजपत्रमा\s+प्रकाशन\s+भएको\s+"
    r"मिति(?:देखि|मा)\s+प्रारम्भ\s+हुनेछ"
)

ORDINAL_DAYS: dict[str, int] = {
    "पहिलो": 1,
    "दोस्रो": 2,
    "तेस्रो": 3,
    "चौथो": 4,
    "पाँचौं": 5,
    "छैटौं": 6,
    "सातौं": 7,
    "आठौं": 8,
    "नवौं": 9,
    "दशौं": 10,
    "एघारौं": 11,
    "बाह्रौं": 12,
    "तेह्रौं": 13,
    "चौधौं": 14,
    "पन्ध्रौं": 15,
    "सोह्रौं": 16,
    "सत्रौं": 17,
    "अठारौं": 18,
    "उन्नाइसौं": 19,
    "बीसौं": 20,
    "एक्काइसौं": 21,
    "बाइसौं": 22,
    "तेइसौं": 23,
    "चौबीसौं": 24,
    "पच्चीसौं": 25,
    "छब्बीसौं": 26,
    "सत्ताइसौं": 27,
    "अठ्ठाइसौं": 28,
    "उनन्तीसौं": 29,
    "तीसौं": 30,
    "एकतीसौँ": 31,
    "एकतिसौँ": 31,
    "बत्तीसौं": 32,
    "चालीसौं": 40,
    "पचासौं": 50,
    "साठीऔं": 60,
    "सत्तरीऔं": 70,
    "असीऔं": 80,
    "नब्बेऔं": 90,
    "एकानब्बेऔं": 91,
    "सयौं": 100,
}


@dataclass(frozen=True)
class CommencementProposal:
    effective_date: Any
    commencement_dependency: str | None
    raw_clause_text: str


def classify_commencement(law: Any, content: str) -> CommencementProposal:
    head = content[:_WINDOW]

    match = _IMMEDIATE_RE.search(head)
    if match:
        if law.enactment_ad is None:
            return CommencementProposal(None, "enactment_date_unknown", match.group(0))
        return CommencementProposal(law.enactment_ad, None, match.group(0))

    match = _RELATIVE_RE.search(head)
    if match:
        days = ORDINAL_DAYS.get(match.group(1))
        if days is not None and law.enactment_ad is not None:
            return CommencementProposal(
                law.enactment_ad + timedelta(days=days), None, match.group(0)
            )
        return CommencementProposal(None, "unparsed_relative_delay", match.group(0))

    match = _GAZETTE_RE.search(head)
    if match:
        return CommencementProposal(
            None, "gazette_notification_pending", match.group(0)
        )

    match = _PUBLICATION_RE.search(head)
    if match:
        if law.enactment_ad is not None:
            return CommencementProposal(law.enactment_ad, None, match.group(0))
        return CommencementProposal(None, "publication_date_unknown", match.group(0))

    return CommencementProposal(None, "no_commencement_clause", "")


def extract_commencement_proposals(
    *, law: Any, content: str, source_pub_id: str, conn: Any
) -> None:
    proposal = classify_commencement(law, content)
    component_uris = (
        [law.uri]
        if proposal.commencement_dependency == "no_commencement_clause"
        else [component.uri for component in law.components]
    )
    for component_uri in component_uris:
        propose_lifecycle_commence(
            conn,
            component_uri,
            source_pub_id,
            effective_date=proposal.effective_date,
            commencement_dependency=proposal.commencement_dependency,
            raw_clause_text=proposal.raw_clause_text,
        )
