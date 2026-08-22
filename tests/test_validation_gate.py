from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, cast

import app.retrieval.validation_gate as gate


def test_validation_gate_abstains_bad_hash(monkeypatch: Any) -> None:
    monkeypatch.setattr(gate, "_expression", lambda conn, uri, as_of: ("पाठ", "0" * 64))
    monkeypatch.setattr(gate, "eligible_chunk_ids", lambda conn, as_of: {"/c/1"})
    out = gate.validate_and_render(
        [{"claim": "x", "evidence_id": "/c/1"}], date(2024, 1, 1), cast(Any, object())
    )
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


def test_validation_gate_renders_citation(monkeypatch: Any) -> None:
    text = "वैध पाठ"
    monkeypatch.setattr(
        gate,
        "_expression",
        lambda conn, uri, as_of: (text, hashlib.sha256(text.encode()).hexdigest()),
    )
    monkeypatch.setattr(gate, "eligible_chunk_ids", lambda conn, as_of: {"/c/1"})
    monkeypatch.setattr(
        gate,
        "_citation",
        lambda conn, uri, as_of: {"component_uri": uri, "as_of": as_of.isoformat()},
    )
    out = gate.validate_and_render(
        [{"claim": "x", "evidence_id": "/c/1"}], date(2024, 1, 1), cast(Any, object())
    )
    assert out[0]["abstained"] is False
    assert out[0]["citation"]["component_uri"] == "/c/1"


def test_expression_reads_chunks() -> None:
    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def execute(self, sql: str, params: dict[str, Any]) -> None:
            self.sql = sql
            self.params = params

        def fetchone(self) -> tuple[str, str]:
            return ("text", "hash")

    class Conn:
        def __init__(self) -> None:
            self.cursor_obj = Cursor()

        def cursor(self) -> Cursor:
            return self.cursor_obj

    conn = Conn()
    assert gate._expression(
        cast(Any, conn), "00000000-0000-0000-0000-000000000001", date(2024, 1, 1)
    ) == ("text", "hash")
    assert "FROM chunks" in conn.cursor_obj.sql
    assert "span_sha256" in conn.cursor_obj.sql
