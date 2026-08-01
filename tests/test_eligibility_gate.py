from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval.eligibility_gate import is_eligible


class Cursor:
    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.sql = sql
        self.params = params

    def fetchone(self) -> tuple[bool]:
        return (True,)


class Conn:
    def __init__(self) -> None:
        self.cursor_obj = Cursor()

    def cursor(self) -> Cursor:
        return self.cursor_obj


def test_is_eligible_calls_sql_function() -> None:
    conn = Conn()
    assert is_eligible(cast(Any, conn), "/c/1", date(2024, 1, 1))
    assert conn.cursor_obj.params == ("/c/1", date(2024, 1, 1))
