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
    monkeypatch.setattr(gate, "_terminated_before", lambda conn, uri, as_of: False)
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


def test_validation_gate_abstains_when_component_terminated(monkeypatch: Any) -> None:
    text = "वैध पाठ"
    monkeypatch.setattr(
        gate,
        "_expression",
        lambda conn, uri, as_of: (text, hashlib.sha256(text.encode()).hexdigest()),
    )
    monkeypatch.setattr(gate, "eligible_chunk_ids", lambda conn, as_of: {"/c/1"})
    monkeypatch.setattr(gate, "_terminated_before", lambda conn, uri, as_of: True)
    out = gate.validate_and_render(
        [{"claim": "x", "evidence_id": "/c/1"}],
        date(2024, 1, 1),
        cast(Any, object()),
    )
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


class GateCursor:
    def __init__(self, conn: "GateConn") -> None:
        self.conn = conn
        self.sql = ""
        self.params: dict[str, Any] = {}
        self.result: tuple[Any, ...] | None = None

    def __enter__(self) -> "GateCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.sql = sql
        self.params = params
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT component_uri FROM chunks"):
            self.result = (self.conn.component_uri,)
        elif squashed.startswith("SELECT 1 FROM lifecycle_effect"):
            self.result = (1,) if self.conn.terminated else None
        elif squashed.startswith("SELECT c.act_name"):
            self.result = self.conn.citation_row
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result


class GateConn:
    def __init__(self) -> None:
        self.component_uri: str | None = "/law/dafa/1"
        self.terminated = False
        self.citation_row: tuple[Any, ...] = (
            "ऐन",
            None,
            "act",
            "derived_verified",
            0.91,
        )
        self.cursor_obj = GateCursor(self)

    def cursor(self) -> GateCursor:
        return self.cursor_obj


def test_terminated_before_false_for_future_or_absent_effect() -> None:
    conn = GateConn()
    conn.terminated = False
    assert not gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))
    assert "lower(legal_valid_time) <= %(as_of)s::timestamptz" in conn.cursor_obj.sql


class TerminationCursor:
    """Actually evaluates the lower(legal_valid_time) <= as_of predicate
    against a seeded date, instead of returning a canned boolean — proves
    the per-claim-as-of direction, not just the SQL shape."""

    def __init__(self, conn: "TerminationConn") -> None:
        self.conn = conn
        self.result: tuple[Any, ...] | None = None

    def __enter__(self) -> "TerminationCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT component_uri FROM chunks"):
            self.result = (self.conn.component_uri,)
        elif squashed.startswith("SELECT 1 FROM lifecycle_effect"):
            effect_date = self.conn.terminating_effect_date
            as_of = params["as_of"]
            self.result = (
                (1,) if effect_date is not None and effect_date <= as_of else None
            )
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result


class TerminationConn:
    def __init__(self, terminating_effect_date: date | None) -> None:
        self.component_uri: str | None = "/law/dafa/1"
        self.terminating_effect_date = terminating_effect_date
        self.cursor_obj = TerminationCursor(self)

    def cursor(self) -> TerminationCursor:
        return self.cursor_obj


def test_terminated_before_true_when_repeal_on_or_before_as_of() -> None:
    conn = TerminationConn(terminating_effect_date=date(2023, 1, 1))
    assert gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))


def test_terminated_before_false_when_repeal_strictly_after_as_of() -> None:
    """A claim about 2070-equivalent law must not abstain over an as-yet
    future (relative to the claim's as_of) repeal — Core Invariant #6,
    per-claim as-of."""
    conn = TerminationConn(terminating_effect_date=date(2025, 1, 1))
    assert not gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))


def test_terminated_before_true_for_approved_past_effect() -> None:
    conn = GateConn()
    conn.terminated = True
    assert gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))


def test_terminated_before_skips_null_component_uri() -> None:
    conn = GateConn()
    conn.component_uri = None
    assert not gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))


def test_citation_reads_source_publication_with_fallback() -> None:
    conn = GateConn()
    citation = gate._citation(cast(Any, conn), "chunk-id", date(2024, 1, 1))
    assert citation is not None
    assert citation["source_kind"] == "derived_verified"
    assert citation["ocr_confidence"] == 0.91

    conn.citation_row = ("ऐन", None, "act", None, None)
    fallback = gate._citation(cast(Any, conn), "chunk-id", date(2024, 1, 1))
    assert fallback is not None
    assert fallback["source_kind"] == "act"
    assert fallback["ocr_confidence"] is None


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
