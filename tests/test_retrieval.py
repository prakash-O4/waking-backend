from __future__ import annotations

import inspect
import sys
import types
from datetime import date
from typing import Any, cast

import app.retrieval.postgres_retriever as r
import app.retrieval.validation_gate as vg
from app.retrieval.reranker import rerank


ROW1 = ("c1", "text one", "h1", "Act", None, "section", "1", 0.9)
ROW2 = ("c2", "text two", "h2", "Act", None, "section", "2", 0.8)
EXACT_ROW = ("c3", "दफा 94 text", "h3", "Act", None, "section", "94", "doc3")


class Cursor:
    def __init__(
        self,
        vector: list[tuple[Any, ...]],
        lexical: list[tuple[Any, ...]],
        exact: list[tuple[Any, ...]] | None = None,
        works: list[tuple[str, str]] | None = None,
    ) -> None:
        self.vector = vector
        self.lexical = lexical
        self.exact = exact or []
        self.works = works or []
        self.calls = 0
        self.sql = ""
        self.params: dict[str, Any] = {}
        self.sqls: list[str] = []
        self.params_history: list[dict[str, Any]] = []

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.calls += 1
        self.sql = sql
        self.params = params
        self.sqls.append(sql)
        self.params_history.append(params)

    def fetchall(self) -> list[tuple[Any, ...]]:
        if "embedding <=>" in self.sql:
            return self.vector
        if "ts_rank_cd" in self.sql:
            return self.lexical
        if (
            "c.work_id" in self.sql
            or "parent_section" in self.sql
            or "अनुसूची" in self.sql
        ):
            eligible = set(self.params.get("eligible", []))
            return [row for row in self.exact if row[0] in eligible]
        return [
            row[:8] if len(row) == 8 and isinstance(row[7], str) else row[:7]
            for row in [*self.vector, *self.lexical, *self.exact]
        ]

    def fetchone(self) -> tuple[str, str] | None:
        matches = [row for row in self.works if row[1] in self.params["query"]]
        if not matches:
            return None
        return max(matches, key=lambda row: len(row[1]))


class Conn:
    def __init__(
        self,
        vector: list[tuple[Any, ...]],
        lexical: list[tuple[Any, ...]],
        exact: list[tuple[Any, ...]] | None = None,
        works: list[tuple[str, str]] | None = None,
    ) -> None:
        self.cursor_obj = Cursor(vector, lexical, exact, works)

    def cursor(self) -> Cursor:
        return self.cursor_obj


def patch_common(monkeypatch: Any, eligible: set[str]) -> None:
    monkeypatch.setattr(r, "eligible_chunk_ids", lambda conn, as_of: eligible)
    monkeypatch.setattr(r, "translate_query", lambda query: None)
    monkeypatch.setattr(r, "_embed_query", lambda query: [0.1, 0.2])
    monkeypatch.setattr(r, "rerank", lambda query, hits, k: hits[:k])


LLM_METADATA_COLUMNS = ("summary", "keywords", "relevant_questions")


def test_hit_does_not_surface_llm_metadata() -> None:
    hit = r._hit(
        (
            "c1",
            "authoritative text",
            "hash",
            "Act",
            None,
            "section",
            "1",
            "doc-source",
            ["llm keyword"],
            ["llm question"],
        ),
        0.5,
    )
    assert set(hit) == {
        "component_uri",
        "text_ne",
        "text_hash",
        "score",
        "vector_score",
        "work_title_ne",
        "chunk_type",
        "section_number",
        "document_source_id",
    }
    assert "llm keyword" not in repr(hit)
    assert "llm question" not in repr(hit)


def test_retriever_sql_does_not_select_llm_metadata() -> None:
    source = inspect.getsource(r.retrieve_postgres)
    assert not any(column in source for column in LLM_METADATA_COLUMNS)


def test_validation_gate_sql_does_not_read_llm_metadata() -> None:
    source = inspect.getsource(vg)
    assert not any(column in source for column in LLM_METADATA_COLUMNS)


def test_is_devanagari_pure_nepali() -> None:
    assert r._is_devanagari("दफा १ अनुसार") is True


def test_is_devanagari_pure_english() -> None:
    assert r._is_devanagari("what is section 1") is False


def test_is_devanagari_mixed_romanized() -> None:
    assert r._is_devanagari("muluki ain ko dafa") is False


def test_parse_section_reference() -> None:
    assert r._parse_section_reference("दफा 94") == "94"
    assert r._parse_section_reference("धारा 51") == "51"
    assert r._parse_section_reference("उपदफा (2)") is None
    assert r._parse_section_reference("उपदफा (2), दफा 9 अनुसार") == "9"
    assert r._parse_section_reference("ordinary query") is None


def test_parse_subsection_reference() -> None:
    assert r._parse_subsection_reference("उपदफा (2)") == "2"
    assert r._parse_subsection_reference("ordinary query") is None


def test_parse_schedule_reference() -> None:
    assert r._parse_schedule_reference("अनुसूची 3") == "3"
    assert r._parse_schedule_reference("ordinary query") is None


def test_resolve_act_title_longest_match_wins() -> None:
    conn = Conn(
        [],
        [],
        works=[("short", "देवानी संहिता"), ("long", "मुलुकी देवानी संहिता")],
    )

    assert r._resolve_act_title(cast(Any, conn), "मुलुकी देवानी संहिता दफा 1") == "long"
    assert "strpos(%(query)s, title_ne) > 0" in conn.cursor_obj.sql


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


def test_bare_subsection_does_not_run_exact_section_filter(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c3"})
    conn = Conn([], [], exact=[EXACT_ROW])

    out = r.retrieve_postgres(cast(Any, conn), "उपदफा (2) मा के छ?", date(2024, 1, 1))

    assert out == []
    assert not any("num" in params for params in conn.cursor_obj.params_history)


def test_exact_lookup_only_result_survives(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c3"})
    conn = Conn([], [], exact=[EXACT_ROW])
    out = r.retrieve_postgres(
        cast(Any, conn),
        "दफा 94",
        date(2024, 1, 1),
        k=5,
    )

    assert [h["component_uri"] for h in out] == ["c3"]
    assert out[0]["document_source_id"] == "doc3"
    assert any(
        params.get("eligible") == ["c3"] and params.get("num") == "94"
        for params in conn.cursor_obj.params_history
    )


def test_exact_lookup_uses_work_title_filter(monkeypatch: Any) -> None:
    patch_common(monkeypatch, {"c3"})
    conn = Conn([], [], exact=[EXACT_ROW], works=[("work-1", "भन्सार महसुल ऐन २०८१")])

    r.retrieve_postgres(cast(Any, conn), "भन्सार महसुल ऐन २०८१ दफा 94", date(2024, 1, 1))

    assert any(
        params.get("work_id") == "work-1" for params in conn.cursor_obj.params_history
    )


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


def test_cohere_reranks_when_key_set(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = "key"

    class FakeResult:
        index = 1

    class FakeResponse:
        results = [FakeResult()]

    class FakeClient:
        def __init__(self, key: str) -> None:
            pass

        def rerank(self, **kwargs: Any) -> FakeResponse:
            return FakeResponse()

    cohere_mod = types.ModuleType("cohere")
    setattr(cohere_mod, "Client", FakeClient)
    monkeypatch.setitem(sys.modules, "cohere", cohere_mod)
    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b", "reranker_tier": "cohere"}]


def test_flashrank_fallback_when_cohere_raises(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = "key"

    class BadClient:
        def __init__(self, key: str) -> None:
            pass

        def rerank(self, **kwargs: Any) -> None:
            raise RuntimeError("rate limited")

    cohere_mod = types.ModuleType("cohere")
    setattr(cohere_mod, "Client", BadClient)
    monkeypatch.setitem(sys.modules, "cohere", cohere_mod)
    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr(
        "app.retrieval.reranker._flashrank_rerank", lambda q, h, k: [h[1]]
    )

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b", "reranker_tier": "flashrank"}]


def test_flashrank_used_when_no_cohere_key(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = ""

    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr(
        "app.retrieval.reranker._flashrank_rerank", lambda q, h, k: [h[1]]
    )

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b", "reranker_tier": "flashrank"}]


def test_passthrough_when_both_fail(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = ""

    def bad_flashrank(q: str, h: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        raise RuntimeError("model error")

    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr("app.retrieval.reranker._flashrank_rerank", bad_flashrank)

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "a", "reranker_tier": "passthrough"}]
