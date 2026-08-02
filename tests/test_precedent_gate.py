from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.eval.gates import check_overruled_as_good_law
from app.retrieval.precedent_retriever import retrieve_precedent


class Cursor:
    def __init__(
        self,
        fetchone: tuple[Any, ...] | None = None,
        fetchall: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone
        self.fetchall_result = fetchall or []
        self.sql = ""
        self.params: object = None

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: object = None) -> None:
        self.sql = sql
        self.params = params

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.fetchone_result

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.fetchall_result


class Conn:
    def __init__(self, cursors: list[Cursor]) -> None:
        self.cursors = cursors
        self.used: list[Cursor] = []

    def cursor(self) -> Cursor:
        cursor = self.cursors.pop(0)
        self.used.append(cursor)
        return cursor


def gate_conn(is_good_law: bool) -> Conn:
    return Conn(
        [
            Cursor(),
            Cursor(("target",)),
            Cursor(("holding",)),
            Cursor(("source",)),
            Cursor(),
            Cursor((is_good_law,)),
            Cursor(),
        ]
    )


def test_check_overruled_as_good_law_returns_zero_when_blocked() -> None:
    assert check_overruled_as_good_law(cast(Any, gate_conn(False))) == 0


def test_check_overruled_as_good_law_detects_good_law_violation() -> None:
    assert check_overruled_as_good_law(cast(Any, gate_conn(True))) == 1


def test_retrieve_precedent_empty_db_returns_empty_list() -> None:
    conn = Conn([Cursor(fetchall=[])])
    assert retrieve_precedent(cast(Any, conn), "contract", date(2024, 1, 1)) == []


def test_retrieve_precedent_includes_good_law_holding() -> None:
    conn = Conn(
        [
            Cursor(
                fetchall=[("h1", "holding text", "/case/1", "Case 1", date(2020, 1, 1))]
            ),
            Cursor(fetchone=(True,)),
        ]
    )

    assert retrieve_precedent(cast(Any, conn), "holding", date(2024, 1, 1)) == [
        {
            "holding_id": "h1",
            "holding_text": "holding text",
            "case_uri": "/case/1",
            "case_title": "Case 1",
            "decided_date": date(2020, 1, 1),
        }
    ]


def test_retrieve_precedent_excludes_overruled_holding() -> None:
    conn = Conn(
        [
            Cursor(
                fetchall=[("h1", "holding text", "/case/1", "Case 1", date(2020, 1, 1))]
            ),
            Cursor(fetchone=(False,)),
        ]
    )

    assert retrieve_precedent(cast(Any, conn), "holding", date(2024, 1, 1)) == []
