from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, cast

import app.retrieval.validation_gate as gate


def test_validation_gate_abstains_bad_hash(monkeypatch: Any) -> None:
    monkeypatch.setattr(gate, "_expression", lambda conn, uri, as_of: ("पाठ", "0" * 64))
    monkeypatch.setattr(gate, "is_eligible", lambda conn, uri, as_of: True)
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
    monkeypatch.setattr(gate, "is_eligible", lambda conn, uri, as_of: True)
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
