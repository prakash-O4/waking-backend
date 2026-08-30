"""
Tests for the enabling-power co-retrieval node in the query graph.

The eligibility gate must be respected: a repealed enabling section must not be
surfaced as a co-reference.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.retrieval import query_graph as qg


@pytest.fixture
def mock_conn() -> MagicMock:
    return MagicMock()


@pytest.fixture
def base_hit() -> dict[str, Any]:
    return {
        "component_uri": "reg-chunk-1",
        "text_ne": "regulation text",
        "score": 0.9,
        "work_title_ne": "Test Regulation",
        "chunk_type": "दफा १",
        "section_number": "१",
        "document_source_id": "reg-1",
        "_issue_idx": 0,
    }


def _make_config(conn: Any) -> Any:
    return {"configurable": {"conn": conn}}


def test_enabling_chunk_passes_eligibility_gate(
    monkeypatch: pytest.MonkeyPatch,
    mock_conn: MagicMock,
    base_hit: dict[str, Any],
) -> None:
    """If the enabling chunk is not eligible, it is not co-retrieved."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [
        ("sub-work-id",),  # chunk -> subordinate work_id
        ("parent-work-id", "55", "dafa"),  # work_relations link
        ("parent-chunk-1", "parent text", "sha256", "Parent Act", "parent-1"),
    ]
    monkeypatch.setattr(qg, "eligible_chunk_ids", lambda _conn, _as_of: set())

    state: dict[str, Any] = {
        "session_as_of": date(2025, 1, 1),
        "all_hits": [base_hit],
    }
    result = qg.enabling_power_resolver_node(state, _make_config(mock_conn))

    assert result == {}


def test_enabling_resolver_coretrieves(
    monkeypatch: pytest.MonkeyPatch,
    mock_conn: MagicMock,
    base_hit: dict[str, Any],
) -> None:
    """Happy path: a resolved, eligible enabling chunk is appended as [CO-REF]."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [
        ("sub-work-id",),
        ("parent-work-id", "55", "dafa"),
        ("parent-chunk-1", "parent text", "sha256", "Parent Act", "parent-1"),
    ]
    monkeypatch.setattr(
        qg, "eligible_chunk_ids", lambda _conn, _as_of: {"parent-chunk-1"}
    )

    state: dict[str, Any] = {
        "session_as_of": date(2025, 1, 1),
        "all_hits": [base_hit],
    }
    result = qg.enabling_power_resolver_node(state, _make_config(mock_conn))

    assert "all_hits" in result
    assert len(result["all_hits"]) == 2
    original, added = result["all_hits"]
    assert original["component_uri"] == "reg-chunk-1"
    assert added["component_uri"] == "parent-chunk-1"
    assert added["co_retrieved"] is True
    assert added["_issue_idx"] == 0
    assert added["work_title_ne"] == "Parent Act"


def test_enabling_resolver_null_link_skipped(
    mock_conn: MagicMock,
    base_hit: dict[str, Any],
) -> None:
    """A work_relations row with enabling_work_id=NULL produces no co-retrieval."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [
        ("sub-work-id",),
        None,  # no resolved link
    ]

    state: dict[str, Any] = {
        "session_as_of": date(2025, 1, 1),
        "all_hits": [base_hit],
    }
    result = qg.enabling_power_resolver_node(state, _make_config(mock_conn))

    assert result == {}


def test_enabling_resolver_dedupes_duplicate_parents(
    monkeypatch: pytest.MonkeyPatch,
    mock_conn: MagicMock,
    base_hit: dict[str, Any],
) -> None:
    """Multiple regulation hits pointing to the same parent दफा add it only once."""
    second_hit = {**base_hit, "component_uri": "reg-chunk-2"}
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [
        ("sub-work-id",),
        ("parent-work-id", "55", "dafa"),
        ("parent-chunk-1", "parent text", "sha256", "Parent Act", "parent-1"),
        ("sub-work-id",),
        ("parent-work-id", "55", "dafa"),
        ("parent-chunk-1", "parent text", "sha256", "Parent Act", "parent-1"),
    ]
    monkeypatch.setattr(
        qg, "eligible_chunk_ids", lambda _conn, _as_of: {"parent-chunk-1"}
    )

    state: dict[str, Any] = {
        "session_as_of": date(2025, 1, 1),
        "all_hits": [base_hit, second_hit],
    }
    result = qg.enabling_power_resolver_node(state, _make_config(mock_conn))

    assert len(result["all_hits"]) == 3
    assert sum(1 for h in result["all_hits"] if h.get("co_retrieved")) == 1
