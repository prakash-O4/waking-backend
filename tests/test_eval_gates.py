from __future__ import annotations

import inspect
from typing import Any

import app.eval.gates as gates


class Cursor:
    def __init__(self, value: int) -> None:
        self.value = value
        self.sql = ""

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.sql = sql

    def fetchone(self) -> tuple[int]:
        return (self.value,)


class Conn:
    def __init__(self, value: int) -> None:
        self.cursor_obj = Cursor(value)

    def cursor(self) -> Cursor:
        return self.cursor_obj


def test_live_repealed_gate_is_independent_sql() -> None:
    source = inspect.getsource(gates.check_repealed_as_current_live)
    assert "eligible_chunk_ids" not in source
    assert "is_eligible" not in source
    assert "lifecycle_effect" in source


def test_live_pending_gate_is_independent_sql() -> None:
    source = inspect.getsource(gates.check_not_yet_effective_as_current_live)
    assert "eligible_chunk_ids" not in source
    assert "is_eligible" not in source
    assert "lifecycle_effect" in source


def test_live_stale_expression_gate_is_independent_sql() -> None:
    source = inspect.getsource(gates.check_stale_expression_as_current_live)
    assert "eligible_chunk_ids" not in source
    assert "is_expression_current(" not in source
    assert "lifecycle_effect" in source
    assert "lower(le.transaction_time) > c.created_at" in source


def test_live_gates_return_counts() -> None:
    conn = Conn(2)
    assert gates.check_repealed_as_current_live(conn) == 2  # type: ignore[arg-type]
    assert gates.check_not_yet_effective_as_current_live(conn) == 2  # type: ignore[arg-type]
    assert gates.check_stale_expression_as_current_live(conn) == 2  # type: ignore[arg-type]
