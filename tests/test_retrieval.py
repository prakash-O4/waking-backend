from __future__ import annotations

import sys
import types
from datetime import date
from typing import Any, cast

import app.retrieval.postgres_retriever as r
from app.retrieval.reranker import rerank


ROW1 = ("c1", "text one", "h1", "Act", None, "section", "1", 0.9)
ROW2 = ("c2", "text two", "h2", "Act", None, "section", "2", 0.8)


class Cursor:
    def __init__(
        self, vector: list[tuple[Any, ...]], lexical: list[tuple[Any, ...]]
    ) -> None:
        self.vector = vector
        self.lexical = lexical
        self.calls = 0
        self.sql = ""
        self.params: dict[str, Any] = {}

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.calls += 1
        self.sql = sql
        self.params = params

    def fetchall(self) -> list[tuple[Any, ...]]:
        if "embedding <=>" in self.sql:
            return self.vector
        if "ts_rank_cd" in self.sql:
            return self.lexical
        return [row[:7] for row in [*self.vector, *self.lexical]]


class Conn:
    def __init__(
        self, vector: list[tuple[Any, ...]], lexical: list[tuple[Any, ...]]
    ) -> None:
        self.cursor_obj = Cursor(vector, lexical)

    def cursor(self) -> Cursor:
        return self.cursor_obj


def patch_common(monkeypatch: Any, eligible: set[str]) -> None:
    monkeypatch.setattr(r, "eligible_chunk_ids", lambda conn, as_of: eligible)
    monkeypatch.setattr(r, "translate_query", lambda query: None)
    monkeypatch.setattr(r, "_embed_query", lambda query: [0.1, 0.2])
    monkeypatch.setattr(r, "rerank", lambda query, hits, k: hits[:k])


def test_is_devanagari_pure_nepali() -> None:
    assert r._is_devanagari("दफा १ अनुसार") is True


def test_is_devanagari_pure_english() -> None:
    assert r._is_devanagari("what is section 1") is False


def test_is_devanagari_mixed_romanized() -> None:
    assert r._is_devanagari("muluki ain ko dafa") is False


def test_translate_query_skips_devanagari(monkeypatch: Any) -> None:
    class Settings:
        GEMINI_API_KEY = "key"

    module = types.ModuleType("langchain_google_genai")

    class BadLLM:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("LLM should not be called")

    setattr(module, "ChatGoogleGenerativeAI", BadLLM)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    monkeypatch.setattr(r, "get_settings", lambda: Settings())
    assert r.translate_query("दफा १") is None


def test_translate_query_returns_none_on_api_failure(monkeypatch: Any) -> None:
    class Settings:
        GEMINI_API_KEY = "key"

    module = types.ModuleType("langchain_google_genai")

    class BadLLM:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    setattr(module, "ChatGoogleGenerativeAI", BadLLM)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    monkeypatch.setattr(r, "get_settings", lambda: Settings())
    assert r.translate_query("what is section 1") is None


def test_translate_query_skips_when_key_unset(monkeypatch: Any) -> None:
    class Settings:
        GEMINI_API_KEY = ""

    module = types.ModuleType("langchain_google_genai")

    class BadLLM:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("LLM should not be called")

    setattr(module, "ChatGoogleGenerativeAI", BadLLM)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    monkeypatch.setattr(r, "get_settings", lambda: Settings())
    assert r.translate_query("section 1") is None


def test_empty_eligible_set_returns_empty(monkeypatch: Any) -> None:
    called = False

    def embed(query: str) -> list[float]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(r, "eligible_chunk_ids", lambda conn, as_of: set())
    monkeypatch.setattr(r, "translate_query", lambda query: None)
    monkeypatch.setattr(r, "_embed_query", embed)
    assert r.retrieve_postgres(cast(Any, object()), "x", date(2024, 1, 1)) == []
    assert called is False


def test_vector_arm_results_only(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c1"})
    out = r.retrieve_postgres(cast(Any, Conn([ROW1], [])), "law", date(2024, 1, 1), k=5)
    assert [h["component_uri"] for h in out] == ["c1"]
    assert out[0]["work_title_ne"] == "Act"


def test_both_arms_rrf_merges(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c1", "c2"})
    out = r.retrieve_postgres(
        cast(Any, Conn([ROW1, ROW2], [ROW2])), "law", date(2024, 1, 1), k=5
    )
    assert [h["component_uri"] for h in out] == ["c2", "c1"]
    assert len(out) == 2


def test_relevance_gate_returns_empty(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c1"})
    monkeypatch.setattr(r, "_RELEVANCE_THRESHOLD", 1.0)
    assert (
        r.retrieve_postgres(cast(Any, Conn([ROW1], [])), "law", date(2024, 1, 1)) == []
    )


def test_dual_path_uses_four_lists_when_translated(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c1", "c2"})
    monkeypatch.setattr(r, "translate_query", lambda query: "दफा १")
    captured = []

    def rrf(lists: list[list[str]]) -> list[tuple[str, float]]:
        nonlocal captured
        captured = lists
        return [("c1", 1.0), ("c2", 1.0)]

    monkeypatch.setattr(r, "_rrf", rrf)
    r.retrieve_postgres(cast(Any, Conn([ROW1], [ROW2])), "section 1", date(2024, 1, 1))
    assert len(captured) == 4


def test_dual_path_falls_back_to_single_when_translation_none(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c1", "c2"})
    captured = []

    def rrf(lists: list[list[str]]) -> list[tuple[str, float]]:
        nonlocal captured
        captured = lists
        return [("c1", 1.0), ("c2", 1.0)]

    monkeypatch.setattr(r, "_rrf", rrf)
    r.retrieve_postgres(cast(Any, Conn([ROW1], [ROW2])), "section 1", date(2024, 1, 1))
    assert len(captured) == 2


def test_reranker_skipped_when_cohere_key_unset(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = ""

    monkeypatch.setattr("app.config.get_settings", lambda: Settings())
    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "a"}]
