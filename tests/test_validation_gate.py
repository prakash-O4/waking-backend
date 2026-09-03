from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, cast

import app.retrieval.validation_gate as gate

_CHUNK_TEXT = (
    "दफा २९४ अनुसार करदाताले आयकर स्वीकार गर्नुपर्ने हुन्छ " "र तोकिएको म्यादभित्र बुझाउनुपर्छ।"
)
_QUOTE = "करदाताले आयकर स्वीकार गर्नुपर्ने हुन्छ"


def _passing_stubs(monkeypatch: Any, chunk_text: str = _CHUNK_TEXT) -> None:
    monkeypatch.setattr(
        gate,
        "_expression",
        lambda conn, uri, as_of: (
            chunk_text,
            hashlib.sha256(chunk_text.encode()).hexdigest(),
        ),
    )
    monkeypatch.setattr(gate, "eligible_chunk_ids", lambda conn, as_of: {"/c/1"})
    monkeypatch.setattr(gate, "_terminated_before", lambda conn, uri, as_of: False)
    monkeypatch.setattr(
        gate,
        "_citation",
        lambda conn, uri, as_of: {"component_uri": uri, "as_of": as_of.isoformat()},
    )


def _render(claims: list[dict[str, str]]) -> list[dict[str, Any]]:
    return gate.validate_and_render(claims, date(2024, 1, 1), cast(Any, object()))


def test_validation_gate_abstains_bad_hash(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        gate, "_expression", lambda conn, uri, as_of: (_CHUNK_TEXT, "0" * 64)
    )
    monkeypatch.setattr(gate, "eligible_chunk_ids", lambda conn, as_of: {"/c/1"})
    monkeypatch.setattr(gate, "_terminated_before", lambda conn, uri, as_of: False)
    # Quote is a genuine substring: if abstention still happens, it must be the
    # hash check, not claim support, that stopped this claim.
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": _QUOTE}])
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


def test_validation_gate_renders_citation(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": _QUOTE}])
    assert out[0]["abstained"] is False
    assert out[0]["citation"]["component_uri"] == "/c/1"


def test_validation_gate_abstains_when_component_terminated(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    monkeypatch.setattr(gate, "_terminated_before", lambda conn, uri, as_of: True)
    # Quote is a genuine substring: abstention must come from termination.
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": _QUOTE}])
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


def test_claim_support_quote_not_in_chunk_abstains(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    out = _render(
        [
            {
                "claim": "x",
                "evidence_id": "/c/1",
                "quote": "यो वाक्य चंकमा कतै पनि छैन र मेल खाँदैन",
            }
        ]
    )
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


def test_claim_support_normalizes_digits_and_whitespace(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    # ASCII digits + collapsed newlines/spaces vs. Devanagari digits in the chunk.
    quote = "दफा 294 अनुसार\nकरदाताले   आयकर"
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": quote}])
    assert out[0]["abstained"] is False
    assert out[0]["citation"] is not None


def test_claim_support_empty_or_missing_quote_abstains(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    for quote in ({}, {"quote": ""}, {"quote": "   \n  "}):
        claim = {"claim": "x", "evidence_id": "/c/1", **quote}
        out = _render([claim])
        assert out[0]["abstained"] is True
        assert out[0]["citation"] is None


def test_claim_support_short_quote_abstains_despite_substring_match(
    monkeypatch: Any,
) -> None:
    _passing_stubs(monkeypatch)
    # "करदाताले आयकर" is a genuine substring but under the 15-char floor.
    quote = "करदाताले आयकर"
    assert quote in _CHUNK_TEXT
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": quote}])
    assert out[0]["abstained"] is True
    assert out[0]["citation"] is None


def test_claim_support_exact_substring_passes(monkeypatch: Any) -> None:
    _passing_stubs(monkeypatch)
    assert _QUOTE in _CHUNK_TEXT
    assert len(_QUOTE) >= gate._MIN_QUOTE_CHARS
    out = _render([{"claim": "x", "evidence_id": "/c/1", "quote": _QUOTE}])
    assert out[0]["abstained"] is False
    assert out[0]["citation"] is not None


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
        elif squashed.startswith("SELECT is_eligible"):
            self.result = (not self.conn.terminated,)
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


def test_terminated_before_uses_canonical_is_eligible() -> None:
    conn = GateConn()
    conn.terminated = False
    assert not gate._terminated_before(cast(Any, conn), "chunk-id", date(2024, 1, 1))
    assert conn.cursor_obj.sql == "SELECT is_eligible(%s, %s)"


class TerminationCursor:
    """Actually evaluates the canonical is_eligible predicate."""

    def __init__(self, conn: "TerminationConn") -> None:
        self.conn = conn
        self.result: tuple[Any, ...] | None = None

    def __enter__(self) -> "TerminationCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any] | tuple[Any, ...]) -> None:
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT component_uri FROM chunks"):
            self.result = (self.conn.component_uri,)
        elif squashed.startswith("SELECT is_eligible"):
            _component_uri, raw_as_of = params
            as_of = cast(date, raw_as_of)
            commenced = self.conn.commence_date <= as_of
            terminated = (
                self.conn.terminating_effect_date is not None
                and self.conn.terminating_effect_date <= as_of
            )
            self.result = (commenced and not terminated,)
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result


class TerminationConn:
    def __init__(self, terminating_effect_date: date | None) -> None:
        self.component_uri: str | None = "/law/dafa/1"
        self.commence_date = date(2020, 1, 1)
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
