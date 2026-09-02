from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.retrieval import gated_orchestrator as orchestrator
from app.retrieval import query_graph as qg


class CoRetrieveCursor:
    def __init__(
        self,
        parent_by_child: dict[str, str | None],
        parents: dict[str, tuple[Any, ...]],
    ) -> None:
        self.parent_by_child = parent_by_child
        self.parents = parents
        self.rows: list[tuple[Any, ...]] = []
        self.queries = 0

    def __enter__(self) -> "CoRetrieveCursor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def execute(self, _sql: str, params: dict[str, Any]) -> None:
        self.queries += 1
        eligible = set(params["eligible"])
        child_id = str(params["hit_id"])
        parent_id = self.parent_by_child.get(child_id)
        self.rows = (
            [(child_id, parent_id, *self.parents[parent_id])]
            if parent_id and parent_id in eligible
            else []
        )

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None


class CoRetrieveConn:
    def __init__(
        self,
        parent_by_child: dict[str, str | None],
        parents: dict[str, tuple[Any, ...]] | None = None,
    ) -> None:
        self.cursor_obj = CoRetrieveCursor(
            parent_by_child,
            parents
            or {
                "parent-1": (
                    "parent text",
                    "sha256",
                    "Parent Act",
                    None,
                    "दफा १८",
                    "१८",
                    "src-1",
                )
            },
        )

    def cursor(self) -> CoRetrieveCursor:
        return self.cursor_obj


def _hit(**extra: Any) -> dict[str, Any]:
    return {
        "component_uri": "child-1",
        "text_ne": "स्पष्टीकरण text",
        "text_hash": "child-hash",
        "score": 0.9,
        "work_title_ne": "Child Act",
        "chunk_type": "proviso",
        "section_number": "१८",
        "document_source_id": "src-1",
        "_issue_idx": 2,
        **extra,
    }


def test_co_retrieve_parent_adds_eligible_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator, "eligible_chunk_ids", lambda _conn, _as_of: {"parent-1"}
    )
    conn = CoRetrieveConn({"child-1": "parent-1"})

    added = orchestrator._resolve_co_retrieve_parents([_hit()], date(2025, 1, 1), conn)

    assert added == [
        {
            "component_uri": "parent-1",
            "text_ne": "parent text",
            "text_hash": "sha256",
            "score": 0.0,
            "work_title_ne": "Parent Act",
            "chunk_type": "दफा १८",
            "section_number": "१८",
            "document_source_id": "src-1",
            "co_retrieved": True,
            "_issue_idx": 2,
        }
    ]


def test_co_retrieve_parent_null_link_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator, "eligible_chunk_ids", lambda _conn, _as_of: {"parent-1"}
    )

    added = orchestrator._resolve_co_retrieve_parents(
        [_hit()], date(2025, 1, 1), CoRetrieveConn({"child-1": None})
    )

    assert added == []


def test_co_retrieve_parent_ineligible_parent_does_not_drop_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "eligible_chunk_ids", lambda _conn, _as_of: set())
    hit = _hit()
    state: dict[str, Any] = {"session_as_of": date(2025, 1, 1), "all_hits": [hit]}

    result = qg.co_retrieve_parent_resolver_node(
        state, {"configurable": {"conn": CoRetrieveConn({"child-1": "parent-1"})}}
    )

    assert result == {}
    assert state["all_hits"] == [hit]


def test_co_retrieve_parent_respects_hit_order_when_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_by_child: dict[str, str | None] = {
        f"child-{i}": f"parent-{i}" for i in range(7)
    }
    parents = {
        f"parent-{i}": (
            f"parent text {i}",
            f"sha256-{i}",
            f"Parent Act {i}",
            None,
            f"दफा {i}",
            str(i),
            f"src-{i}",
        )
        for i in range(7)
    }
    monkeypatch.setattr(
        orchestrator, "eligible_chunk_ids", lambda _conn, _as_of: set(parents)
    )
    hits = [_hit(component_uri=f"child-{i}", _issue_idx=i) for i in range(7)]

    added = orchestrator._resolve_co_retrieve_parents(
        hits, date(2025, 1, 1), CoRetrieveConn(parent_by_child, parents)
    )

    assert [h["component_uri"] for h in added] == [
        "parent-0",
        "parent-1",
        "parent-2",
        "parent-3",
        "parent-4",
    ]
    assert [h["_issue_idx"] for h in added] == [0, 1, 2, 3, 4]


def test_co_retrieve_parent_skips_already_coretrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator, "eligible_chunk_ids", lambda _conn, _as_of: {"parent-1"}
    )
    conn = CoRetrieveConn({"child-1": "parent-1"})

    added = orchestrator._resolve_co_retrieve_parents(
        [_hit(co_retrieved=True)], date(2025, 1, 1), conn
    )

    assert added == []
    assert conn.cursor_obj.queries == 0


def test_build_graph_orders_co_retrieve_before_cross_ref() -> None:
    edges = {(edge.source, edge.target) for edge in qg.build_graph().get_graph().edges}

    assert ("authority_ranker", "co_retrieve_parent_resolver") in edges
    assert ("co_retrieve_parent_resolver", "cross_ref_resolver") in edges
    assert ("authority_ranker", "cross_ref_resolver") not in edges
