from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from app.authority.writer import upsert_expression


@dataclass
class _Component:
    uri: str
    text_ne: str
    text_hash: str


class _Cursor:
    def __init__(self, conn: "_Conn") -> None:
        self.conn = conn

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT 1 FROM expression"):
            component_uri, text_hash = params
            self._found = (component_uri, text_hash) in self.conn.existing
        elif squashed.startswith("INSERT INTO expression"):
            component_uri, _as_of, _text_ne, text_hash = params
            self.conn.existing.add((component_uri, text_hash))
            self.conn.inserts += 1
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[int] | None:
        return (1,) if self._found else None


class _Conn:
    def __init__(self) -> None:
        self.existing: set[tuple[str, str]] = set()
        self.inserts = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def test_upsert_expression_skips_duplicate_on_rerun_with_different_as_of() -> None:
    """A rerun on a later date must not duplicate a row for unchanged text —
    the bug that let backfill_authority_layer.py double the expression
    table when it ran with as_of=today() against rows ingested on an
    earlier date."""
    conn = _Conn()
    component = _Component(
        uri="/np/act/2063/demo/dafa/1", text_ne="पाठ", text_hash="a" * 64
    )

    upsert_expression(cast(Any, conn), cast(Any, component), as_of=date(2026, 8, 30))
    assert conn.inserts == 1

    upsert_expression(cast(Any, conn), cast(Any, component), as_of=date(2026, 9, 1))
    assert conn.inserts == 1


def test_upsert_expression_inserts_new_row_on_real_text_change() -> None:
    conn = _Conn()
    original = _Component(
        uri="/np/act/2063/demo/dafa/1", text_ne="पाठ", text_hash="a" * 64
    )
    amended = _Component(
        uri="/np/act/2063/demo/dafa/1", text_ne="नयाँ पाठ", text_hash="b" * 64
    )

    upsert_expression(cast(Any, conn), cast(Any, original), as_of=date(2026, 8, 30))
    upsert_expression(cast(Any, conn), cast(Any, amended), as_of=date(2026, 9, 1))
    assert conn.inserts == 2
