from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator

from fastapi.testclient import TestClient

import app.main as main
import app.retrieval.query_graph as query_graph


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
    def fake_answer(
        question: str, as_of: object, conn: object, user_id: str | None = None
    ) -> dict[str, Any]:
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


def test_ask_stream_delegates_to_orchestrator(monkeypatch: Any) -> None:
    def fake_stream_answer(
        question: str, as_of: object, conn: object, user_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        yield {"stage": "retrieve", "status": "done", "latency_ms": 1}
        yield {
            "stage": "final",
            "status": "done",
            "response": {"results": [{"claim": question}]},
        }

    monkeypatch.setattr(main, "stream_answer", fake_stream_answer)

    res = client(monkeypatch).post(
        "/ask/stream",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 200
    assert 'data: {"stage": "retrieve", "status": "done", "latency_ms": 1}' in res.text
    assert (
        'data: {"stage": "final", "status": "done", "response": {"results": [{"claim": "q"}]}}'
        in res.text
    )


def test_ask_stream_quota_still_blocks(monkeypatch: Any) -> None:
    class QuotaSupabaseHelper(FakeSupabaseHelper):
        def check_daily_quota(self, user_id: str) -> bool:
            return True

    monkeypatch.setattr(main, "SupabaseHelper", QuotaSupabaseHelper)

    res = TestClient(main.app).post(
        "/ask/stream",
        headers={"Authorization": "Bearer token"},
        json={"question": "q", "as_of": "2024-01-01"},
    )

    assert res.status_code == 404


def test_stream_query_does_not_leak_non_final_delta(monkeypatch: Any) -> None:
    class FakeGraph:
        def stream(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
            yield {"retrieve": {"all_hits": ["raw law text"]}}
            yield {"answer_composer": {"_response": {"ok": True}, "secret": "nope"}}

    monkeypatch.setattr(query_graph, "_graph", FakeGraph())
    monkeypatch.setattr(query_graph, "_get_lf_client", lambda: None)

    events = list(query_graph.stream_query("q", date(2024, 1, 1), object()))

    assert events[0].keys() == {"stage", "status", "latency_ms"}
    assert events[0]["stage"] == "retrieve"
    assert "all_hits" not in events[0]
    assert events[1] == {"stage": "final", "status": "done", "response": {"ok": True}}


def test_stream_query_final_matches_run_query(monkeypatch: Any) -> None:
    response = {"results": [{"claim": "q"}]}

    class FakeGraph:
        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"_response": response}

        def stream(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
            yield {"answer_composer": {"_response": response}}

    monkeypatch.setattr(query_graph, "_graph", FakeGraph())
    monkeypatch.setattr(query_graph, "_get_lf_client", lambda: None)

    expected = query_graph.run_query("q", date(2024, 1, 1), object())
    final = list(query_graph.stream_query("q", date(2024, 1, 1), object()))[-1]

    assert final["response"] == expected
