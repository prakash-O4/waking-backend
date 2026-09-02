from __future__ import annotations

from typing import Any

from scripts.ingest_laws import _print_parent_child_coverage


class _Cursor:
    def __init__(self, rows: list[tuple[str, int, int, int]]) -> None:
        self.rows = rows
        self.sql = ""

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.sql = sql

    def fetchall(self) -> list[tuple[str, int, int, int]]:
        return self.rows


class _Conn:
    def __init__(self, rows: list[tuple[str, int, int, int]]) -> None:
        self.cur = _Cursor(rows)

    def cursor(self) -> _Cursor:
        return self.cur


def test_print_parent_child_coverage_flags_link_expected_orphans(capsys: Any) -> None:
    conn = _Conn(
        [
            ("subsection", 17501, 0, 17501),
            ("proviso", 51, 50, 1),
        ]
    )

    _print_parent_child_coverage(conn)

    out = capsys.readouterr().out
    assert "subsection" in out
    assert "proviso" in out
    assert "⚠ expected linked" in out
    assert "parent_section IS NOT NULL" in conn.cur.sql
