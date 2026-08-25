from __future__ import annotations

from datetime import date
from types import ModuleType, SimpleNamespace
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
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": question, "as_of": as_of, "work_type_hint": None}
            ],
        },
    )
    monkeypatch.setattr(orchestrator, "retrieve_postgres", retrieve)
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits: {
            "claims": [
                {
                    "claim": "ok",
                    "evidence_id": "/law/1",
                    "issue": "test",
                    "applicability": "high",
                    "condition": None,
                }
            ]
        },
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
        "_fact_extract",
        lambda question, as_of: {
            "facts": {"parties": [], "events": [], "dates": [], "location": None},
            "missing_facts": [],
            "issue_queries": [
                {"query": "old", "as_of": date(2020, 1, 1), "work_type_hint": None},
                {"query": "new", "as_of": date(2024, 1, 1), "work_type_hint": None},
            ],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits: {
            "claims": [
                {
                    "claim": issue_queries[0]["query"],
                    "evidence_id": hits[0]["component_uri"],
                    "issue": "test",
                    "applicability": "high",
                    "condition": None,
                }
            ]
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
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": "first", "as_of": as_of, "work_type_hint": None},
                {"query": "second", "as_of": as_of, "work_type_hint": None},
            ],
        },
    )
    times = iter([0.0, 0.0, orchestrator.WALL_CLOCK_CAP + 0.1])
    monkeypatch.setattr(
        "app.retrieval.gated_orchestrator.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits: {
            "claims": [
                {
                    "claim": issue_queries[0]["query"],
                    "evidence_id": hits[0]["component_uri"],
                    "issue": "test",
                    "applicability": "high",
                    "condition": None,
                }
            ]
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


def test_graph_compiles_and_returns_expected_shape(monkeypatch: Any) -> None:
    """Graph wires correctly and answer() returns the right response shape."""
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": question, "as_of": as_of, "work_type_hint": None}
            ],
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5: [
            {"component_uri": "/law/1", "text_ne": "text"}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits: {
            "claims": [
                {
                    "claim": "ok",
                    "evidence_id": "/law/1",
                    "issue": "test",
                    "applicability": "high",
                    "condition": None,
                }
            ]
        },
    )
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert "as_of" in body
    assert "query_type" in body
    assert "abstained" in body
    assert "results" in body
    assert body["results"][0]["claim"] == "ok"


def test_query_state_schema_complete() -> None:
    """QueryState TypedDict has all required Stage 1+ fields."""
    import typing

    from app.retrieval.query_state import QueryState

    keys = set(typing.get_type_hints(QueryState).keys())
    required = {
        "raw_query",
        "session_as_of",
        "subqueries",
        "all_hits",
        "all_results",
        "query_type",
        "wall_clock_start",
        "facts",
        "missing_facts",
        "issue_queries",
        "interrupted",
        "interrupt_prompt",
        "_pending_results",
    }
    assert required.issubset(keys)


def test_fact_extract_success_populates_issue_queries(monkeypatch: Any) -> None:
    """_fact_extract parses Gemini response and returns normalised issue_queries."""
    import json

    class FakeResp:
        content = json.dumps(
            {
                "facts": {
                    "parties": ["landlord"],
                    "events": ["eviction"],
                    "dates": [],
                    "location": None,
                },
                "missing_facts": [{"fact": "written agreement?", "type": "clarifying"}],
                "issue_queries": [
                    {
                        "query": "भाडा सम्झौता सम्बन्धी कानून",
                        "as_of": "2024-01-01",
                        "work_type_hint": "Act",
                    },
                    {
                        "query": "घर खाली गराउने प्रक्रिया",
                        "as_of": None,
                        "work_type_hint": None,
                    },
                ],
            }
        )

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any) -> FakeResp:
            return FakeResp()

    langchain_google_genai = ModuleType("langchain_google_genai")
    setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeLLM)
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_google_genai", langchain_google_genai
    )
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            GEMINI_API_KEY="key",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )

    result = orchestrator._fact_extract("eviction question", date(2024, 1, 1))

    assert result["facts"]["parties"] == ["landlord"]
    assert result["missing_facts"][0]["type"] == "clarifying"
    assert len(result["issue_queries"]) == 2
    assert result["issue_queries"][0]["as_of"] == date(2024, 1, 1)
    assert result["issue_queries"][1]["as_of"] == date(
        2024, 1, 1
    )  # null → session_as_of


def test_fact_extract_failure_returns_single_query_fallback(monkeypatch: Any) -> None:
    """_fact_extract falls back to single raw query when Gemini is unavailable."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(GEMINI_API_KEY="", LANGFUSE_PUBLIC_KEY=""),
    )

    result = orchestrator._fact_extract("what is the notice period?", date(2024, 6, 1))

    assert result["facts"] is None
    assert result["missing_facts"] == []
    assert len(result["issue_queries"]) == 1
    assert result["issue_queries"][0]["query"] == "what is the notice period?"
    assert result["issue_queries"][0]["as_of"] == date(2024, 6, 1)


def test_authority_rank_hits_sorts_by_tier() -> None:
    """Hits reordered by tier ASC, score DESC; tier attached to each hit."""

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            pass

        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("chunk-act", "Act", "act"),
                ("chunk-const", "Constitution", "act"),
                ("chunk-reg", "Rule", "regulation"),
            ]

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {
            "component_uri": "chunk-act",
            "score": 0.8,
            "section_number": "45",
            "text_ne": "",
        },
        {
            "component_uri": "chunk-const",
            "score": 0.6,
            "section_number": "3",
            "text_ne": "",
        },
        {
            "component_uri": "chunk-reg",
            "score": 0.9,
            "section_number": "12",
            "text_ne": "",
        },
    ]

    result = orchestrator._authority_rank_hits(hits, _Conn())

    assert result[0]["component_uri"] == "chunk-const"
    assert result[0]["tier"] == 1
    assert result[1]["component_uri"] == "chunk-act"
    assert result[1]["tier"] == 2
    assert result[2]["component_uri"] == "chunk-reg"
    assert result[2]["tier"] == 3


def test_authority_rank_hits_conflict_flag() -> None:
    """Same section_number covered by lower-tier chunk gets conflict_flag=True."""

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            pass

        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("chunk-act", "Act", "act"),
                ("chunk-rule", "Rule", "regulation"),
            ]

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {
            "component_uri": "chunk-act",
            "score": 0.8,
            "section_number": "10",
            "text_ne": "",
        },
        {
            "component_uri": "chunk-rule",
            "score": 0.9,
            "section_number": "10",
            "text_ne": "",
        },
    ]

    result = orchestrator._authority_rank_hits(hits, _Conn())

    act_hit = next(h for h in result if h["component_uri"] == "chunk-act")
    rule_hit = next(h for h in result if h["component_uri"] == "chunk-rule")
    assert "conflict_flag" not in act_hit
    assert rule_hit.get("conflict_flag") is True


def test_authority_rank_hits_failure_returns_unchanged() -> None:
    """Exception from conn → original hits returned unchanged."""
    hits = [{"component_uri": "x", "score": 0.5, "section_number": "", "text_ne": ""}]
    result = orchestrator._authority_rank_hits(hits, object())
    assert result is hits


def test_resolve_cross_refs_finds_section_reference(monkeypatch: Any) -> None:
    """दफा reference in text → co-retrieved chunk added with co_retrieved=True."""
    monkeypatch.setattr(
        orchestrator,
        "eligible_chunk_ids",
        lambda conn, as_of: {"chunk-100", "chunk-456"},
    )

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            self._p = params

        def fetchone(self) -> tuple[Any, ...] | None:
            if self._p.get("section_num") == "456":
                return (
                    "chunk-456",
                    "दफा ४५६ को पाठ",
                    "hash456",
                    "Act Name",
                    None,
                    "dafa",
                    "456",
                    "src-001",
                )
            return None

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {
            "component_uri": "chunk-100",
            "text_ne": "यो दफा ४५६ मा उल्लेख भएको छ",
            "score": 0.8,
            "section_number": "100",
            "document_source_id": "src-001",
        }
    ]

    result = orchestrator._resolve_cross_refs(hits, date(2024, 1, 1), _Conn())

    assert len(result) == 1
    assert result[0]["component_uri"] == "chunk-456"
    assert result[0]["co_retrieved"] is True
    assert result[0]["section_number"] == "456"


def test_resolve_cross_refs_failure_returns_empty() -> None:
    """Exception from conn → empty list returned."""
    hits = [
        {
            "component_uri": "chunk-1",
            "text_ne": "दफा ४५ को प्रावधान",
            "score": 0.5,
            "section_number": "1",
            "document_source_id": "src-1",
        }
    ]
    result = orchestrator._resolve_cross_refs(hits, date(2024, 1, 1), object())
    assert result == []


def test_structured_claims_success(monkeypatch: Any) -> None:
    """_structured_claims calls AzureChatOpenAI and parses the JSON response."""
    import json as _json

    class FakeResp:
        content = _json.dumps(
            {
                "claims": [
                    {
                        "claim": "भाडावाला लाई ३५ दिनको सूचना दिनुपर्छ",
                        "evidence_id": "chunk-abc",
                        "issue": "eviction_notice",
                        "applicability": "high",
                        "condition": "written agreement exists",
                    }
                ],
                "abstain": False,
            }
        )

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any, config: Any = None) -> FakeResp:
            return FakeResp()

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", FakeLLM)
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            AZURE_OPENAI_LLM_KEY="key",
            AZURE_OPENAI_LLM_ENDPOINT="https://example.openai.azure.com/",
            AZURE_OPENAI_LLM_DEPLOYMENT="gpt-4.1-mini",
            AZURE_OPENAI_API_VERSION="2023-05-15",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )

    hits = [
        {"component_uri": "chunk-abc", "text_ne": "दफा ३५", "tier": 2, "score": 0.9}
    ]
    issue_queries = [
        {"query": "भाडा सम्बन्धी कानून", "as_of": date(2024, 1, 1), "work_type_hint": None}
    ]

    result = orchestrator._structured_claims(None, issue_queries, hits)

    assert result is not None
    assert result["abstain"] is False
    assert result["claims"][0]["evidence_id"] == "chunk-abc"
    assert result["claims"][0]["applicability"] == "high"


def test_structured_claims_no_key_returns_none(monkeypatch: Any) -> None:
    """_structured_claims returns None immediately when AZURE_OPENAI_LLM_KEY is unset."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(AZURE_OPENAI_LLM_KEY=""),
    )

    result = orchestrator._structured_claims(
        None,
        [{"query": "q", "as_of": date(2024, 1, 1), "work_type_hint": None}],
        [{"component_uri": "c", "text_ne": "text", "tier": 2, "score": 0.5}],
    )

    assert result is None
