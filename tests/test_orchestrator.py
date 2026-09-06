from __future__ import annotations

from datetime import date
from itertools import chain, repeat
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

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
        conn: object, query: str, as_of: date, k: int = 5, **kwargs: Any
    ) -> list[dict[str, str]]:
        seen.append(as_of)
        return [{"component_uri": "/law/1", "text_ne": "text"}]

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
    monkeypatch.setattr(orchestrator, "retrieve_postgres", retrieve)
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits, **kw: {
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
    monkeypatch.setattr(orchestrator, "_compose_answer", lambda *a, **kw: None)
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
        lambda question, as_of, **kw: {
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
        lambda conn, query, as_of, k=5, **kw: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits, **kw: {
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

    monkeypatch.setattr(orchestrator, "_compose_answer", lambda *a, **kw: None)
    monkeypatch.setattr(orchestrator, "validate_and_render", validate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["query_type"] == "complex"
    assert validate_as_ofs == [date(2020, 1, 1), date(2024, 1, 1)]
    assert [r["as_of"] for r in body["results"]] == ["2020-01-01", "2024-01-01"]


def test_wall_clock_cap_returns_validated_so_far(monkeypatch: Any) -> None:
    import app.retrieval.query_graph as query_graph

    # This test freezes time.monotonic() globally (see below). A real Langfuse
    # client's background thread relies on real elapsed time for its own
    # queueing/backoff and hangs when fed a frozen clock, so keep tracing
    # disabled here regardless of the real LANGFUSE_PUBLIC_KEY in the env.
    monkeypatch.setattr(query_graph, "_get_lf_client", lambda: None)
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of, **kw: {
            "facts": None,
            "missing_facts": [],
            "issue_queries": [
                {"query": "first", "as_of": as_of, "work_type_hint": None},
                {"query": "second", "as_of": as_of, "work_type_hint": None},
            ],
        },
    )
    # Yield the wall-clock cap once, then continue returning a large value so
    # code using a real generator-backed clock does not raise StopIteration.
    times = chain(
        [0.0, 0.0, orchestrator.WALL_CLOCK_CAP + 0.1],
        repeat(orchestrator.WALL_CLOCK_CAP + 0.1),
    )
    monkeypatch.setattr(
        "app.retrieval.gated_orchestrator.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5, **kw: [
            {"component_uri": f"/{query}", "text_ne": query}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits, **kw: {
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
    monkeypatch.setattr(orchestrator, "_compose_answer", lambda *a, **kw: None)
    monkeypatch.setattr(orchestrator, "validate_and_render", _validating_gate)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert [r["claim"] for r in body["results"]] == ["first"]


def test_graph_compiles_and_returns_expected_shape(monkeypatch: Any) -> None:
    """Graph wires correctly and answer() returns the right response shape."""
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
        orchestrator,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5, **kw: [
            {"component_uri": "/law/1", "text_ne": "text"}
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_structured_claims",
        lambda facts, issue_queries, hits, **kw: {
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
    monkeypatch.setattr(orchestrator, "_compose_answer", lambda *a, **kw: None)
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
        content = [
            {
                "type": "text",
                "text": "```json\n" + json.dumps(
                    {
                        "facts": {
                            "parties": ["landlord"],
                            "events": ["eviction"],
                            "dates": [],
                            "location": None,
                        },
                        "missing_facts": [
                            {"fact": "written agreement?", "type": "clarifying"}
                        ],
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
                ) + "\n```",
                "extras": {},
            }
        ]

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any, config: Any = None) -> FakeResp:
            return FakeResp()

    langchain_openai = ModuleType("langchain_openai")
    setattr(langchain_openai, "AzureChatOpenAI", FakeLLM)
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_openai", langchain_openai
    )
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            AZURE_OPENAI_LLM_KEY="key",
            AZURE_OPENAI_LLM_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_LLM_DEPLOYMENT="gpt-4.1-mini",
            AZURE_OPENAI_API_VERSION="2023-05-15",
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
    """_fact_extract falls back to single raw query when the LLM is unavailable."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(AZURE_OPENAI_LLM_KEY="", LANGFUSE_PUBLIC_KEY=""),
    )

    result = orchestrator._fact_extract("what is the notice period?", date(2024, 6, 1))

    assert result["facts"] is None
    assert result["missing_facts"] == []
    assert len(result["issue_queries"]) == 1
    assert result["issue_queries"][0]["query"] == "what is the notice period?"
    assert result["issue_queries"][0]["as_of"] == date(2024, 6, 1)


def test_fact_extract_logs_api_failure(monkeypatch: Any) -> None:
    class BadLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any, config: Any = None) -> Any:
            raise RuntimeError("boom")

    langchain_openai = ModuleType("langchain_openai")
    setattr(langchain_openai, "AzureChatOpenAI", BadLLM)
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_openai", langchain_openai
    )
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            AZURE_OPENAI_LLM_KEY="key",
            AZURE_OPENAI_LLM_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_LLM_DEPLOYMENT="gpt-4.1-mini",
            AZURE_OPENAI_API_VERSION="2023-05-15",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )
    warnings: list[str] = []
    monkeypatch.setattr(orchestrator.logger, "warning", warnings.append)

    result = orchestrator._fact_extract("what is the notice period?", date(2024, 6, 1))

    assert result["issue_queries"][0]["query"] == "what is the notice period?"
    assert warnings and "_fact_extract failed: boom" in warnings[0]



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


def test_compose_answer_success(monkeypatch: Any) -> None:
    """_compose_answer calls Gemini and returns ADR Node 7 format dict."""
    import json as _json

    class FakeResp:
        content = [
            {
                "type": "text",
                "text": (
                    "```json\n"
                    + _json.dumps(
                        {
                            "relevant_sections": [
                                {
                                    "section": "Muluki Dewani Samhita, दफा 456",
                                    "why_applicable": "governs residential tenancy notice period",
                                    "applicability": "high",
                                    "condition": "written tenancy agreement exists",
                                    "citation": {},
                                }
                            ],
                            "missing_facts": ["Is there a written tenancy agreement?"],
                            "conflicts": [],
                            "plain_language": "घर बहालमा लिनेलाई ३५ दिनको सूचना दिनुपर्छ।",
                            "disclaimer": "यो कानुनी जानकारी हो, कानुनी सल्लाह होइन।",
                            "as_of": "2024-01-01",
                            "abstained": False,
                        }
                    )
                    + "\n```"
                ),
                "extras": {},
            }
        ]

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any, config: Any = None) -> FakeResp:
            return FakeResp()

    langchain_openai = ModuleType("langchain_openai")
    setattr(langchain_openai, "AzureChatOpenAI", FakeLLM)
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_openai", langchain_openai
    )
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            AZURE_OPENAI_LLM_KEY="key",
            AZURE_OPENAI_LLM_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_LLM_DEPLOYMENT="gpt-4.1-mini",
            AZURE_OPENAI_API_VERSION="2023-05-15",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )

    all_results = [
        {
            "claim": "भाडावाला लाई ३५ दिनको सूचना",
            "evidence_id": "chunk-abc",
            "abstained": False,
            "citation": {},
            "as_of": "2024-01-01",
            "issue": "eviction_notice",
            "applicability": "high",
            "condition": None,
        }
    ]
    result = orchestrator._compose_answer(
        None,
        [{"fact": "Is there a written agreement?", "type": "clarifying"}],
        all_results,
        [],
        date(2024, 1, 1),
    )

    assert result is not None
    assert result["abstained"] is False
    assert len(result["relevant_sections"]) == 1
    assert result["relevant_sections"][0]["applicability"] == "high"
    assert "plain_language" in result


def test_revalidate_composed_replaces_model_citation_and_drops_bad_sections() -> None:
    good_citation = {"source": "server"}
    composed = {
        "relevant_sections": [
            {
                "section": "good",
                "evidence_id": "e1",
                "as_of": "2024-01-01",
                "citation": {"source": "model"},
            },
            {"section": "fabricated", "evidence_id": "nope", "as_of": "2024-01-01"},
            {"section": "abstained", "evidence_id": "e2", "as_of": "2024-01-01"},
            "junk",
        ],
        "plain_language": "keep on partial strip",
        "abstained": True,
    }
    all_results = [
        {
            "evidence_id": "e1",
            "as_of": "2024-01-01",
            "abstained": False,
            "citation": good_citation,
        },
        {
            "evidence_id": "e2",
            "as_of": "2024-01-01",
            "abstained": True,
            "citation": {"source": "should-not-render"},
        },
    ]

    result = orchestrator._revalidate_composed(composed, all_results)

    assert result["abstained"] is False
    assert result["plain_language"] == "keep on partial strip"
    assert [s["section"] for s in result["relevant_sections"]] == ["good"]
    assert result["relevant_sections"][0]["citation"] is good_citation


def test_revalidate_composed_keys_by_evidence_id_and_as_of() -> None:
    old = {"source": "old"}
    new = {"source": "new"}

    result = orchestrator._revalidate_composed(
        {
            "relevant_sections": [
                {"section": "new", "evidence_id": "same", "as_of": "2024-01-01"}
            ],
            "plain_language": "ok",
        },
        [
            {
                "evidence_id": "same",
                "as_of": "2020-01-01",
                "abstained": False,
                "citation": old,
            },
            {
                "evidence_id": "same",
                "as_of": "2024-01-01",
                "abstained": False,
                "citation": new,
            },
        ],
    )

    assert result["relevant_sections"][0]["citation"] is new


def test_revalidate_composed_abstains_and_blanks_when_no_sections_survive() -> None:
    result = orchestrator._revalidate_composed(
        {
            "relevant_sections": {"not": "a list"},
            "plain_language": "unsafe prose",
            "abstained": False,
        },
        [],
    )

    assert result["relevant_sections"] == []
    assert result["abstained"] is True
    assert result["plain_language"] == ""


def test_answer_composer_node_revalidates_composed_output(monkeypatch: Any) -> None:
    from app.retrieval import query_graph

    citation = {"source": "server"}
    monkeypatch.setattr(
        query_graph._orch,
        "_compose_answer",
        lambda *args, **kw: {
            "relevant_sections": [
                {
                    "section": "ok",
                    "evidence_id": "e1",
                    "as_of": "2024-01-01",
                    "citation": {"source": "model"},
                }
            ],
            "plain_language": "ok",
            "abstained": True,
        },
    )
    monkeypatch.setattr(
        query_graph._orch, "_emit_answer_trace_from_state", lambda *a: None
    )

    response = query_graph.answer_composer_node(
        {
            "facts": None,
            "missing_facts": [],
            "all_results": [
                {
                    "evidence_id": "e1",
                    "as_of": "2024-01-01",
                    "abstained": False,
                    "citation": citation,
                }
            ],
            "all_hits": [],
            "raw_query": "q",
            "session_as_of": date(2024, 1, 1),
            "query_type": "simple",
            "wall_clock_start": 0.0,
        },
        {"configurable": {}},
    )["_response"]

    assert response["abstained"] is False
    assert response["relevant_sections"][0]["citation"] is citation


def test_answer_composer_node_fallback_abstains_when_all_results_abstain(
    monkeypatch: Any,
) -> None:
    from app.retrieval import query_graph

    monkeypatch.setattr(query_graph._orch, "_compose_answer", lambda *args, **kw: None)
    monkeypatch.setattr(
        query_graph._orch, "_emit_answer_trace_from_state", lambda *a: None
    )

    response = query_graph.answer_composer_node(
        {
            "facts": None,
            "missing_facts": [],
            "all_results": [{"abstained": True}],
            "all_hits": [],
            "raw_query": "q",
            "session_as_of": date(2024, 1, 1),
            "query_type": "simple",
            "wall_clock_start": 0.0,
        },
        {"configurable": {}},
    )["_response"]

    assert response["abstained"] is True


def test_compose_answer_no_key_returns_none(monkeypatch: Any) -> None:
    """_compose_answer returns None immediately when AZURE_OPENAI_LLM_KEY is unset."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(AZURE_OPENAI_LLM_KEY=""),
    )

    result = orchestrator._compose_answer(
        None,
        [],
        [{"claim": "ok", "evidence_id": "x", "abstained": False, "citation": {}}],
        [],
        date(2024, 1, 1),
    )

    assert result is None


def test_compose_answer_logs_api_failure(monkeypatch: Any) -> None:
    class BadLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any, config: Any = None) -> Any:
            raise RuntimeError("boom")

    langchain_openai = ModuleType("langchain_openai")
    setattr(langchain_openai, "AzureChatOpenAI", BadLLM)
    monkeypatch.setitem(
        __import__("sys").modules, "langchain_openai", langchain_openai
    )
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(
            AZURE_OPENAI_LLM_KEY="key",
            AZURE_OPENAI_LLM_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_LLM_DEPLOYMENT="gpt-4.1-mini",
            AZURE_OPENAI_API_VERSION="2023-05-15",
            LANGFUSE_PUBLIC_KEY="",
        ),
    )
    warnings: list[str] = []
    monkeypatch.setattr(orchestrator.logger, "warning", warnings.append)

    result = orchestrator._compose_answer(
        None,
        [],
        [{"claim": "ok", "evidence_id": "x", "abstained": False, "citation": {}}],
        [],
        date(2024, 1, 1),
    )

    assert result is None
    assert warnings and "_compose_answer failed: boom" in warnings[0]



def test_required_missing_fact_returns_interrupted_response(monkeypatch: Any) -> None:
    """When the coverage probe finds no law, required missing facts interrupt."""
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of, **kw: {
            "facts": None,
            "missing_facts": [
                {
                    "fact": "Is this a residential or commercial tenancy?",
                    "type": "required",
                }
            ],
            "issue_queries": [
                {"query": question, "as_of": as_of, "work_type_hint": None}
            ],
        },
    )

    retrieve_called: list[Any] = []

    def retrieve(
        conn: object, q: str, as_of: date, k: int = 5, **kwargs: Any
    ) -> list[Any]:
        retrieve_called.append(1)
        return []

    monkeypatch.setattr(orchestrator, "retrieve_postgres", retrieve)

    conn: Any = object()
    body = orchestrator.answer("q", date(2024, 1, 1), conn)

    assert body["interrupted"] is True
    assert "Is this a residential or commercial tenancy?" in (
        body["interrupt_prompt"] or ""
    )
    assert body["results"] == []
    assert retrieve_called == [1]


def test_emit_trace_uses_vector_score(monkeypatch: Any) -> None:
    """_emit_answer_trace_from_state uses vector_score (not RRF score) in output."""
    captured: dict[str, Any] = {}

    class FakeTrace:
        def update(self, output: Any = None, **kwargs: Any) -> None:
            if output:
                captured.update(output)

        def end(self) -> None:
            pass

    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(LANGFUSE_LOG_CONTENT=False),
    )

    hits = [
        {"score": 0.016, "vector_score": 0.71},
        {"score": 0.015, "vector_score": 0.65},
    ]
    orchestrator._emit_answer_trace_from_state(
        FakeTrace(), "q", date(2024, 1, 1), "simple", [], hits, 0.0
    )

    assert captured["top_chunk_scores"][0] == pytest.approx(0.71)
    assert captured["top_chunk_scores"][1] == pytest.approx(0.65)
    assert captured["gate_decision"] == "abstained"


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "raw_query": "q",
        "session_as_of": date(2024, 1, 1),
        "subqueries": [],
        "all_hits": [],
        "all_results": [],
        "query_type": "simple",
        "wall_clock_start": 0.0,
        "facts": None,
        "missing_facts": [],
        "issue_queries": [],
        "interrupted": False,
        "interrupt_prompt": None,
        "_pending_results": [],
        "_response": {},
    }
    base.update(overrides)
    return base


def test_required_missing_fact_with_probe_hit_becomes_clarifying(
    monkeypatch: Any,
) -> None:
    import app.retrieval.query_graph as qg

    monkeypatch.setattr(
        qg._orch,
        "_fact_extract",
        lambda question, as_of, **kw: {
            "facts": None,
            "missing_facts": [{"fact": "lease type?", "type": "required"}],
            "issue_queries": [
                {"query": question, "as_of": as_of, "work_type_hint": None}
            ],
        },
    )
    monkeypatch.setattr(qg._orch, "retrieve_postgres", lambda *a, **kw: [{"id": "hit"}])
    monkeypatch.setattr(qg._orch, "_wall_clock_expired", lambda start: False)

    out = qg.fact_extractor_node(
        cast(Any, _state()), cast(Any, {"configurable": {"conn": object()}})
    )

    assert out["interrupted"] is False
    assert out["missing_facts"] == [{"fact": "lease type?", "type": "clarifying"}]


def test_co_retrievers_use_issue_as_of(monkeypatch: Any) -> None:
    import app.retrieval.query_graph as qg

    seen_parent: list[date] = []
    seen_cross: list[date] = []

    def parent(
        hits: list[dict[str, Any]], as_of: date, conn: object
    ) -> list[dict[str, Any]]:
        seen_parent.append(as_of)
        return []

    def cross(
        hits: list[dict[str, Any]], as_of: date, conn: object
    ) -> list[dict[str, Any]]:
        seen_cross.append(as_of)
        return []

    monkeypatch.setattr(qg._orch, "_resolve_co_retrieve_parents", parent)
    monkeypatch.setattr(qg._orch, "_resolve_cross_refs", cross)
    state = _state(
        issue_queries=[
            {"query": "old", "as_of": date(2020, 1, 1)},
            {"query": "new", "as_of": date(2024, 1, 1)},
        ],
        all_hits=[
            {"component_uri": "a", "_issue_idx": 0},
            {"component_uri": "b", "_issue_idx": 1},
        ],
    )

    qg.co_retrieve_parent_resolver_node(
        cast(Any, state), cast(Any, {"configurable": {"conn": object()}})
    )
    qg.cross_ref_resolver_node(
        cast(Any, state), cast(Any, {"configurable": {"conn": object()}})
    )

    assert seen_parent == [date(2020, 1, 1), date(2024, 1, 1)]
    assert seen_cross == [date(2020, 1, 1), date(2024, 1, 1)]


def test_enabling_resolver_uses_hit_issue_as_of(monkeypatch: Any) -> None:
    import app.retrieval.query_graph as qg

    seen: list[date] = []

    def fetch(conn: object, hit: dict[str, Any], as_of: date) -> None:
        seen.append(as_of)
        return None

    monkeypatch.setattr(qg, "_fetch_enabling_chunk", fetch)
    state = _state(
        issue_queries=[
            {"query": "old", "as_of": date(2020, 1, 1)},
            {"query": "new", "as_of": date(2024, 1, 1)},
        ],
        all_hits=[
            {"component_uri": "a", "_issue_idx": 0},
            {"component_uri": "b", "_issue_idx": 1},
        ],
    )

    qg.enabling_power_resolver_node(
        cast(Any, state), cast(Any, {"configurable": {"conn": object()}})
    )

    assert seen == [date(2020, 1, 1), date(2024, 1, 1)]


def test_parallel_retrieve_preserves_issue_order_and_closes_pool(
    monkeypatch: Any,
) -> None:
    import app.retrieval.db_pool as db_pool
    import app.retrieval.query_graph as qg

    class Pool:
        closed = False

        def getconn(self) -> object:
            return object()

        def putconn(self, conn: object) -> None:
            pass

        def closeall(self) -> None:
            self.closed = True

    pool = Pool()
    monkeypatch.setattr(db_pool, "make_retrieval_pool", lambda maxconn: pool)
    monkeypatch.setattr(qg._orch, "_wall_clock_expired", lambda start: False)
    monkeypatch.setattr(
        qg._orch,
        "retrieve_postgres",
        lambda conn, query, as_of, k=5, **kw: [
            {"component_uri": query, "text_ne": query}
        ],
    )
    state = _state(
        issue_queries=[
            {"query": "first", "as_of": date(2024, 1, 1)},
            {"query": "second", "as_of": date(2024, 1, 1)},
        ]
    )

    out = qg.retrieve_node(
        cast(Any, state), cast(Any, {"configurable": {"conn": object()}})
    )

    assert [h["component_uri"] for h in out["all_hits"]] == ["first", "second"]
    assert pool.closed is True


def test_answer_response_exposes_degraded_mode(monkeypatch: Any) -> None:
    import app.retrieval.query_graph as qg

    monkeypatch.setattr(qg._orch, "_compose_answer", lambda *a, **kw: None)
    monkeypatch.setattr(
        qg._orch, "_emit_answer_trace_from_state", lambda *a, **kw: None
    )
    state = _state(
        query_type="extractive",
        all_hits=[{"component_uri": "a", "reranker_tier": "flashrank"}],
        all_results=[{"claim": "x", "abstained": False}],
    )

    out = qg.answer_composer_node(cast(Any, state), cast(Any, {"configurable": {}}))

    assert out["_response"]["degraded_mode"] == [
        "reasoner_fallback:extractive",
        "reranker_fallback:flashrank",
    ]
