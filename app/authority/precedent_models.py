from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class RelationType(str, Enum):
    OVERRULES = "overrules"
    REVERSES = "reverses"
    DISTINGUISHES = "distinguishes"
    AFFIRMS = "affirms"
    QUESTIONS = "questions"


@dataclass
class PrecedentRelation:
    source_case_uri: str
    target_holding_id: str
    relation_type: RelationType
    bench_strength: int
    legal_valid_from: date
    source_span: str | None
    approval_status: str = "pending"
