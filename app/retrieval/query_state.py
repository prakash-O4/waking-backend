from __future__ import annotations

from datetime import date
from typing import Any, Optional
from typing_extensions import TypedDict


class QueryState(TypedDict):
    raw_query: str
    session_as_of: date
    subqueries: list[dict[str, Any]]
    all_hits: list[dict[str, Any]]
    all_results: list[dict[str, Any]]
    query_type: str
    wall_clock_start: float

    facts: Any
    missing_facts: list[dict[str, Any]]
    issue_queries: list[dict[str, Any]]
    interrupted: bool
    interrupt_prompt: Optional[str]
    _pending_results: list[dict[str, Any]]
    _response: dict[str, Any]
