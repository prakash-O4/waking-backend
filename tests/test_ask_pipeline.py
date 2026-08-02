from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from fastapi.testclient import TestClient

import app.main as main


class FakeSupabaseHelper:
    def get_user_id(self, token: str | None) -> str:
        return "user-1"

    def check_daily_quota(self, user_id: str) -> bool:
        return False


@contextmanager
def fake_connect() -> Iterator[object]:
    yield object()


def client(monkeypatch: Any) -> TestClient:
    monkeypatch.setattr(main, "SupabaseHelper", FakeSupabaseHelper)
    monkeypatch.setattr(main, "connect", fake_connect)
    return TestClient(main.app)


def test_ask_delegates_to_orchestrator(monkeypatch: Any) -> None:
    def fake_answer(question: str, as_of: object, conn: object) -> dict[str, Any]:
        return {
            "as_of": "2024-01-01",
            "query_type": "simple",
            "abstained": False,
            "results": [{"claim": question, "as_of": "2024-01-01"}],
        }

    monkeypatch.setattr(main, "orchestrator_answer", fake_answer)

    res = client(monkeypatch).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    assert res.json()["results"][0]["claim"] == "q"


def test_ask_quota_still_blocks(monkeypatch: Any) -> None:
    class QuotaSupabaseHelper(FakeSupabaseHelper):
        def check_daily_quota(self, user_id: str) -> bool:
            return True

    monkeypatch.setattr(main, "SupabaseHelper", QuotaSupabaseHelper)

    res = TestClient(main.app).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 404
