from __future__ import annotations

import inspect
from datetime import date
from typing import Any, cast

import app.retrieval.eligibility_gate as eg
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


class FilteringCursor(Cursor):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__([])
        self.input_rows = rows

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        super().execute(sql, params)
        as_of = params["as_of"]
        self.rows = [
            (row["id"],)
            for row in self.input_rows
            if row["ingestion_status"] == "approved"
            and row["source_type"] != "nkp_case"
            and row["effective_date_ad"] is not None
            and row["effective_date_ad"] <= as_of
        ]


class FilteringConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.cursor_obj = FilteringCursor(rows)

    def cursor(self) -> FilteringCursor:
        return self.cursor_obj


def test_eligible_chunk_ids_sql_does_not_read_llm_metadata() -> None:
    source = inspect.getsource(eg.eligible_chunk_ids)
    assert not any(
        column in source for column in ("summary", "keywords", "relevant_questions")
    )


def test_eligible_chunk_ids_excludes_pending_valid_document() -> None:
    conn = Conn([])
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()
    assert "ingestion_status = 'approved'" in conn.cursor_obj.sql
    assert "pending" not in conn.cursor_obj.sql
    assert "quarantined" not in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_quarantined() -> None:
    conn = Conn([])
    out = eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1))
    assert out == set()
    assert "ingestion_status = 'approved'" in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_future_effective_date() -> None:
    conn = Conn([])
    out = eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1))
    assert out == set()
    assert "c.effective_date_ad IS NOT NULL" in conn.cursor_obj.sql
    assert "c.effective_date_ad <= %(as_of)s" in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_nkp_cases() -> None:
    conn = FilteringConn(
        [
            {
                "id": "case-chunk",
                "ingestion_status": "approved",
                "source_type": "nkp_case",
                "effective_date_ad": date(2020, 1, 1),
            },
            {
                "id": "law-chunk",
                "ingestion_status": "approved",
                "source_type": "act",
                "effective_date_ad": date(2020, 1, 1),
            },
        ]
    )
    out = eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1))
    assert out == {"law-chunk"}
    assert "c.source_type <> 'nkp_case'" in conn.cursor_obj.sql


def test_eligible_chunk_ids_excludes_null_effective_date() -> None:
    conn = FilteringConn(
        [
            {
                "id": "undated-law",
                "ingestion_status": "approved",
                "source_type": "act",
                "effective_date_ad": None,
            }
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()


def test_eligible_chunk_ids_empty_result() -> None:
    assert eligible_chunk_ids(cast(Any, Conn([])), date(2024, 1, 1)) == set()
