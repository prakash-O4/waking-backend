from __future__ import annotations

from datetime import date
from typing import Any, cast

import app.retrieval.postgres_retriever as r
from app.retrieval.eligibility_gate import eligible_chunk_ids
from tests.stress.helpers import StressConn, chunk, effect
from tests.test_retrieval import Conn, Cursor, ROW1, patch_common


class FilteringCursor(Cursor):
    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = cast(list[tuple[Any, ...]], super().fetchall())
        if "eligible" not in self.params:
            return rows
        eligible = set(self.params["eligible"])
        return [row for row in rows if row[0] in eligible]


class FilteringConn(Conn):
    def __init__(self) -> None:
        self.cursor_obj = FilteringCursor([ROW1], [ROW1])


def test_romanized_retrieval_cannot_bypass_eligibility(monkeypatch: Any) -> None:
    assert r._is_devanagari("muluki ain ko dafa ek") is False
    # Start from the normal retrieval test patch, then replace the canned set
    # with the same lifecycle predicate used by the gate.
    patch_common(monkeypatch, {"c1"})
    lifecycle = StressConn(
        {
            "c1": chunk(
                "/law/repealed",
                [
                    effect("commence", date(2020, 1, 1)),
                    effect("repeal", date(2024, 1, 1)),
                ],
            )
        }
    )
    monkeypatch.setattr(
        r,
        "eligible_chunk_ids",
        lambda _conn, as_of: eligible_chunk_ids(cast(Any, lifecycle), as_of),
    )
    monkeypatch.setattr(r, "translate_query", lambda _query: "दफा १")

    assert (
        r.retrieve_postgres(
            cast(Any, FilteringConn()), "muluki ain ko dafa ek", date(2024, 1, 1)
        )
        == []
    )
