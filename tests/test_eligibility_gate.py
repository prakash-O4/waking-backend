from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval.eligibility_gate import eligible_chunk_ids


class Cursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self.rows = rows
        self.sql = ""
        self.params: dict[str, Any] = {}

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.sql = sql
        self.params = params

    def fetchall(self) -> list[tuple[str]]:
        return self.rows


class Conn:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self.cursor_obj = Cursor(rows)

    def cursor(self) -> Cursor:
        return self.cursor_obj


def test_eligible_chunk_ids_includes_pending_valid_document() -> None:
    conn = Conn([("chunk-1",)])
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == {"chunk-1"}
    assert "pending" in conn.cursor_obj.sql
    assert "quarantined" not in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_quarantined() -> None:
    conn = Conn([])
    out = eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1))
    assert out == set()
    assert "ingestion_status IN ('approved', 'pending')" in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_future_effective_date() -> None:
    conn = Conn([])
    out = eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1))
    assert out == set()
    assert "c.effective_date_ad <= %(as_of)s" in conn.cursor_obj.sql


def test_eligible_chunk_ids_empty_result() -> None:
    assert eligible_chunk_ids(cast(Any, Conn([])), date(2024, 1, 1)) == set()
