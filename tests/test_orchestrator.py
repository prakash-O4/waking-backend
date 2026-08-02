from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from app.retrieval import gated_orchestrator as orchestrator


def _validating_gate(
    claims: list[dict[str, str]], as_of: date, conn: object
) -> list[dict[str, Any]]:
    return [
        {
            "claim": claim["claim"],
            "evidence_id": claim["evidence_id"],
            "abstained": False,
            "citation": {},
        }
        for claim in claims
    ]


def test_simple_query_uses_single_session_as_of(monkeypatch: Any) -> None:
    seen: list[date] = []

    def retrieve(
        conn: object, query: str, as_of: date, k: int = 5
    ) -> list[dict[str, str]]:
        seen.append(as_of)
        return [{"component_uri": "/law/1", "text_ne": "text"}]

    monkeypatch.setattr(
        orchestrator,
        "_classify_and_decompose",
        lambda question, as_of: [{"subquery": question, "as_of": as_of}],
    )
    monkeypatch.setattr(orchestrator, "_try_retrieve", retrieve)
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {"claims": [{"claim": "ok", "evidence_id": "/law/1"}]},
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "simple"
    assert seen == [date(2024, 1, 1)]
    assert body["results"][0]["as_of"] == "2024-01-01"


def test_complex_query_validates_each_subquery_as_of(monkeypatch: Any) -> None:
    validate_as_ofs: list[date] = []
    monkeypatch.setattr(
        orchestrator,
        "_classify_and_decompose",
        lambda question, as_of: [
            {"subquery": "old", "as_of": date(2020, 1, 1)},
            {"subquery": "new", "as_of": date(2024, 1, 1)},
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_try_retrieve",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {
            "claims": [{"claim": question, "evidence_id": hits[0]["component_uri"]}]
        },
    )

    def validate(
        claims: list[dict[str, str]], as_of: date, conn: object
    ) -> list[dict[str, Any]]:
        validate_as_ofs.append(as_of)
        return _validating_gate(claims, as_of, conn)

    monkeypatch.setattr(orchestrator, "validate_and_render", validate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "complex"
    assert validate_as_ofs == [date(2020, 1, 1), date(2024, 1, 1)]
    assert [r["as_of"] for r in body["results"]] == ["2020-01-01", "2024-01-01"]


def test_wall_clock_cap_returns_validated_so_far(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_classify_and_decompose",
        lambda question, as_of: [
            {"subquery": "first", "as_of": as_of},
            {"subquery": "second", "as_of": as_of},
        ],
    )
    times = iter([0.0, 0.0, orchestrator.WALL_CLOCK_CAP + 0.1])
    monkeypatch.setattr(
        "app.retrieval.gated_orchestrator.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr(
        orchestrator,
        "_try_retrieve",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_model_claims",
        lambda question, hits: {
            "claims": [{"claim": question, "evidence_id": hits[0]["component_uri"]}]
        },
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert [r["claim"] for r in body["results"]] == ["first"]


def test_classifier_failure_falls_back_to_simple(monkeypatch: Any) -> None:
    import langchain_openai

    class FailingChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
            raise RuntimeError("down")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FailingChatOpenAI)

    subqueries = orchestrator._classify_and_decompose("q", date(2024, 1, 1))

    assert subqueries == [{"subquery": "q", "as_of": date(2024, 1, 1)}]
