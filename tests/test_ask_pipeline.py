from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from fastapi.testclient import TestClient

import app.main as main


class FakeSupabaseHelper:
    def get_user_id(self, token: str | None) -> str:
        return "user-1"

    def check_daily_quota(self, user_id: str) -> bool:
        return False


class FakeChatOpenAI:
    content = '{"claims": [{"claim": "valid", "evidence_id": "/law/1/sec/1"}]}'

    def __init__(self, **kwargs: Any) -> None:
        pass

    def invoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace(content=self.content)


@contextmanager
def fake_connect() -> Iterator[object]:
    yield object()


def client(
    monkeypatch: Any, chat_openai: type[FakeChatOpenAI] = FakeChatOpenAI
) -> TestClient:
    monkeypatch.setattr(main, "SupabaseHelper", FakeSupabaseHelper)
    monkeypatch.setattr(main, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(main, "connect", fake_connect)
    return TestClient(main.app)


def test_ask_happy_path(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda query, as_of, k: [{"component_uri": "/law/1/sec/1", "text_ne": "text"}],
    )
    monkeypatch.setattr(
        main,
        "validate_and_render",
        lambda claims, as_of, conn: [
            {
                "claim": claims[0]["claim"],
                "abstained": False,
                "citation": {
                    "component_uri": "/law/1/sec/1",
                    "work_title_ne": "ऐन",
                    "work_title_en": "Act",
                    "as_of": as_of.isoformat(),
                    "source_kind": "official_copy_unverified",
                    "ocr_confidence": None,
                },
            }
        ],
    )

    res = client(monkeypatch).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["as_of"] == "2024-01-01"
    assert body["abstained"] is False
    assert body["results"][0]["citation"]["component_uri"] == "/law/1/sec/1"


def test_ask_abstains_when_no_hits(monkeypatch: Any) -> None:
    monkeypatch.setattr(main, "retrieve", lambda query, as_of, k: [])

    res = client(monkeypatch).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    assert res.json() == {"as_of": "2024-01-01", "abstained": True, "results": []}


def test_ask_abstains_when_model_abstains(monkeypatch: Any) -> None:
    class AbstainingChatOpenAI(FakeChatOpenAI):
        content = '{"abstain": true, "claims": []}'

    monkeypatch.setattr(
        main,
        "retrieve",
        lambda query, as_of, k: [{"component_uri": "/law/1/sec/1", "text_ne": "text"}],
    )

    res = client(monkeypatch, AbstainingChatOpenAI).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    assert res.json() == {"as_of": "2024-01-01", "abstained": True, "results": []}


def test_ask_returns_validation_abstained_claim(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda query, as_of, k: [{"component_uri": "/law/1/sec/1", "text_ne": "text"}],
    )
    monkeypatch.setattr(
        main,
        "validate_and_render",
        lambda claims, as_of, conn: [
            {
                "claim": claims[0]["claim"],
                "evidence_id": claims[0]["evidence_id"],
                "abstained": True,
                "citation": None,
            }
        ],
    )

    res = client(monkeypatch).post(
        "/ask",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["abstained"] is False
    assert body["results"][0]["abstained"] is True
