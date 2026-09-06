from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator

import psycopg2
from fastapi.testclient import TestClient

import app.main as main
from app.retrieval import gated_orchestrator as orchestrator


class FakeSupabaseHelper:
    def get_user_id(self, token: str | None) -> str:
        return "user-1"

    def check_daily_quota(self, user_id: str) -> bool:
        return False


@contextmanager
def fake_connect() -> Iterator[object]:
    yield object()


def test_postgres_down_returns_503(monkeypatch: Any) -> None:
    def down_connect() -> Any:
        raise psycopg2.OperationalError("down")

    monkeypatch.setattr(main, "SupabaseHelper", FakeSupabaseHelper)
    monkeypatch.setattr(main, "connect", down_connect)

    res = TestClient(main.app).post(
        "/ask", json={"question": "q", "as_of": "2024-01-01"}
    )

    assert res.status_code == 503
    assert res.json() == {
        "message": "Authority store unavailable. No validated answers can be provided.",
        "retry_after": 60,
    }


def test_model_down_uses_extractive_claim_still_validated(monkeypatch: Any) -> None:
    hit = {"component_uri": "/law/1", "text_ne": "abcdef" * 100}
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of, **kw: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": question, "as_of": as_of, "work_type_hint": None}
            ],
        },
    )
    monkeypatch.setattr(
        orchestrator, "retrieve_postgres", lambda conn, q, a, k=5, **kw: [hit]
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits, **kw: None,
    )
    monkeypatch.setattr(orchestrator, "_compose_answer", lambda *a, **kw: None)

    def validate(
        claims: list[dict[str, str]], as_of: date, conn: object
    ) -> list[dict[str, Any]]:
        seen["claims"] = claims
        return [{"claim": claims[0]["claim"], "abstained": False, "citation": {}}]

    monkeypatch.setattr(orchestrator, "validate_and_render", validate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "extractive"
    assert len(seen["claims"][0]["claim"]) == orchestrator.EXTRACTIVE_CHARS
    assert body["results"][0]["as_of"] == "2024-01-01"
