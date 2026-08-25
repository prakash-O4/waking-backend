# AGENT-5 — Answer Composer + Missing-Facts Interrupt (Stage 5)

**Branch:** `agent/stage-5-answer-composer`
**Base:** `dev`
**Engineer:** Pi
**ADR:** `docs/adr-001-multi-agent-query-architecture.md` §Node 7 — Answer Composer, §Missing facts handling

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No "Generated with Claude" line. No AI attribution of any kind.

---

## Objective

Complete the 5-stage multi-agent pipeline with two additions:

1. **`answer_composer_node`** — replaces `assemble_node`. Calls Gemini 2.5 Flash to compose a structured final answer from validated claims, surfacing `why_applicable`, `conflicts`, `missing_facts` prompts, and a `plain_language` summary. Degrades gracefully: API failure → return raw validated claims in the current format (unchanged client contract).

2. **Missing-facts interrupt** — `fact_extractor_node` detects `required` missing facts and sets `interrupted=True` + `interrupt_prompt`. A conditional edge routes interrupted requests directly to `answer_composer_node` (which returns the interrupt response immediately, skipping retrieval).

Also included: remove the dead `_classify_and_decompose` function and its test.

New graph (7 nodes, conditional routing):
```
fact_extractor ──→ retrieve → authority_ranker → cross_ref_resolver → reasoner → validate ──→ answer_composer → END
               ↘ (interrupted=True)                                                          ↗
                ─────────────────────────────────────────────────────────────────────────────
```

---

## Acceptance criteria

- `make test` — **66 tests passing**, 2 skipped (64 current − 1 removed + 3 new)
- `make lint` — clean
- `make eval-gates` — zero-tolerance gates unchanged: `repealed-as-current=0`, `not-yet-effective-as-current=0`, `overruled-as-good-law=0`
- Interrupt path: `interrupted=True` in response when any missing fact has `type == "required"`; `retrieve_postgres` NOT called on interrupt path
- Composer failure path: response has `results`, `as_of`, `query_type`, `abstained` (same keys as today)
- `_classify_and_decompose` removed; `import os` removed (no other usage after removal)

---

## Scope — exactly these three files

1. `app/retrieval/gated_orchestrator.py`
2. `app/retrieval/query_graph.py`
3. `tests/test_orchestrator.py`

**Do NOT modify** `query_state.py`, `validation_gate.py`, `config.py`, or any other file.

---

## Part 1 — `gated_orchestrator.py` changes

### 1a. Remove `import os` (line 5)

`os` is only used in `_classify_and_decompose` (being removed). Without it the import is unused and lint will fail.

### 1b. Remove `_classify_and_decompose` entirely

The function starts at the line `def _classify_and_decompose(question: str, session_as_of: date)` and ends with `return [{"subquery": question, "as_of": session_as_of}]`. Delete the whole function. It was replaced by `_fact_extract` in AGENT-2 and has been dead code since then.

### 1c. Add `_compose_answer` (insert after `_structured_claims`, before `_classify_and_decompose` location)

```python
def _compose_answer(
    facts: Any,
    missing_facts: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    conflict_hits: list[dict[str, Any]],
    session_as_of: date,
) -> dict[str, Any] | None:
    """Compose structured final answer using Gemini 2.5 Flash.

    Returns ADR Node 7 format dict on success, None on any failure.
    Failure mode: caller falls back to returning raw validated claims.
    """
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return None

    claims_text = json.dumps(all_results, ensure_ascii=False, default=str)
    facts_text = json.dumps(facts, ensure_ascii=False) if facts else "null"
    missing_text = json.dumps(
        [mf for mf in missing_facts if mf.get("type") in ("clarifying", "informational")],
        ensure_ascii=False,
    )
    conflicts_text = json.dumps(conflict_hits, ensure_ascii=False, default=str)

    system = (
        "You are Wakil-G, a Nepali legal assistant. "
        "Compose a structured legal answer from the provided validated claims. "
        "Never invent law. Never modify citations. Use only what the claims provide. "
        "Output JSON only:\n"
        "{\n"
        '  "relevant_sections": [\n'
        '    {"section": "<law name + दफा number>", "why_applicable": "<reason>",\n'
        '     "applicability": "high|medium|low", "condition": "<condition or null>",\n'
        '     "citation": {}}\n'
        "  ],\n"
        '  "missing_facts": ["<user-facing question about clarifying fact>"],\n'
        '  "conflicts": ["<description of conflict between sources>"],\n'
        '  "plain_language": "<plain Nepali explanation, 2-4 sentences>",\n'
        '  "disclaimer": "यो कानुनी जानकारी हो, कानुनी सल्लाह होइन।",\n'
        f'  "as_of": "{session_as_of.isoformat()}",\n'
        '  "abstained": false\n'
        "}\n"
        "If all claims are abstained or there are no claims: "
        '{"abstained": true, "relevant_sections": [], "missing_facts": [], '
        '"conflicts": [], "plain_language": "", '
        '"disclaimer": "यो कानुनी जानकारी हो, कानुनी सल्लाह होइन।", '
        f'"as_of": "{session_as_of.isoformat()}"' + "}"
    )
    user = (
        f"VALIDATED CLAIMS:\n{claims_text}\n\n"
        f"EXTRACTED FACTS:\n{facts_text}\n\n"
        f"MISSING FACTS (clarifying/informational only):\n{missing_text}\n\n"
        f"CONFLICTS (same section, different authority tier):\n{conflicts_text}"
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=s.GEMINI_API_KEY,
            temperature=0.0,
        )
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return cast(dict[str, Any], json.loads(str(resp.content).strip()))
    except Exception:
        return None
```

---

## Part 2 — `query_graph.py` changes

### 2a. Update `fact_extractor_node` — detect required missing facts

Replace the current body with:

```python
def fact_extractor_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    result = _orch._fact_extract(state["raw_query"], state["session_as_of"])
    issue_queries = result["issue_queries"]
    missing_facts = result["missing_facts"]

    required = [mf for mf in missing_facts if mf.get("type") == "required"]
    interrupted = bool(required)
    interrupt_prompt: str | None = None
    if required:
        interrupt_prompt = (
            "To answer your question I need to know: "
            + "; ".join(mf.get("fact", "") for mf in required)
        )

    return {
        "facts": result["facts"],
        "missing_facts": missing_facts,
        "issue_queries": issue_queries,
        "query_type": "simple" if len(issue_queries) == 1 else "complex",
        "interrupted": interrupted,
        "interrupt_prompt": interrupt_prompt,
    }
```

### 2b. Remove `assemble_node` entirely

Delete the function. It is replaced by `answer_composer_node`.

### 2c. Add `answer_composer_node` (insert before `build_graph`)

```python
def answer_composer_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    """Compose final answer. Returns interrupt response immediately if interrupted."""
    session_as_of = state["session_as_of"]
    query_type = state["query_type"]

    if state.get("interrupted"):
        return {
            "_response": {
                "as_of": session_as_of.isoformat(),
                "query_type": query_type,
                "abstained": False,
                "results": [],
                "interrupted": True,
                "interrupt_prompt": state.get("interrupt_prompt"),
            }
        }

    all_results = state["all_results"]
    all_hits = state["all_hits"]
    raw_query = state["raw_query"]

    _orch._emit_answer_trace_from_state(
        raw_query,
        session_as_of,
        query_type,
        all_results,
        all_hits,
        state["wall_clock_start"],
    )

    conflict_hits = [
        {
            "component_uri": h.get("component_uri", ""),
            "work_type": h.get("work_type", ""),
            "tier": h.get("tier"),
            "section_number": h.get("section_number", ""),
        }
        for h in all_hits
        if h.get("conflict_flag")
    ]

    composed = _orch._compose_answer(
        state["facts"],
        state["missing_facts"],
        all_results,
        conflict_hits,
        session_as_of,
    )

    if composed is None:
        return {
            "_response": {
                "as_of": session_as_of.isoformat(),
                "query_type": query_type,
                "abstained": not all_results,
                "results": all_results,
            }
        }

    composed["query_type"] = query_type
    return {"_response": composed}
```

### 2d. Update `build_graph()` — conditional edge + new node name

```python
def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("fact_extractor", fact_extractor_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("authority_ranker", authority_ranker_node)
    builder.add_node("cross_ref_resolver", cross_ref_resolver_node)
    builder.add_node("reasoner", reasoner_node)
    builder.add_node("validate", validate_node)
    builder.add_node("answer_composer", answer_composer_node)

    builder.add_edge(START, "fact_extractor")
    builder.add_conditional_edges(
        "fact_extractor",
        lambda state: "answer_composer" if state.get("interrupted") else "retrieve",
        {"answer_composer": "answer_composer", "retrieve": "retrieve"},
    )
    builder.add_edge("retrieve", "authority_ranker")
    builder.add_edge("authority_ranker", "cross_ref_resolver")
    builder.add_edge("cross_ref_resolver", "reasoner")
    builder.add_edge("reasoner", "validate")
    builder.add_edge("validate", "answer_composer")
    builder.add_edge("answer_composer", END)

    return builder.compile()
```

---

## Part 3 — `tests/test_orchestrator.py` changes

### 3a. Remove `test_classifier_failure_falls_back_to_simple`

Delete the entire function (tests `_classify_and_decompose` which is being removed). **Count: −1.**

### 3b. Existing tests 1-3, 5 need no changes

All mock `_fact_extract` to return `missing_facts: []` → `interrupted=False` → normal path. In the normal path, `answer_composer_node` calls `_compose_answer(...)`. The real `get_settings()` (not mocked) returns `GEMINI_API_KEY=""` → `_compose_answer` returns `None` → fallback format with `results`, `as_of`, `query_type`, `abstained`. All existing assertions still hold. ✓

### 3c. Add 3 new tests (append after `test_structured_claims_no_key_returns_none`)

```python
def test_compose_answer_success(monkeypatch: Any) -> None:
    """_compose_answer calls Gemini and returns ADR Node 7 format dict."""
    import json as _json

    class FakeResp:
        content = _json.dumps(
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
        lambda: SimpleNamespace(GEMINI_API_KEY="key", LANGFUSE_PUBLIC_KEY=""),
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


def test_compose_answer_no_key_returns_none(monkeypatch: Any) -> None:
    """_compose_answer returns None immediately when GEMINI_API_KEY is unset."""
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(GEMINI_API_KEY=""),
    )

    result = orchestrator._compose_answer(
        None,
        [],
        [{"claim": "ok", "evidence_id": "x", "abstained": False, "citation": {}}],
        [],
        date(2024, 1, 1),
    )

    assert result is None


def test_required_missing_fact_returns_interrupted_response(monkeypatch: Any) -> None:
    """When fact_extractor finds a required missing fact, retrieve is skipped and
    the response has interrupted=True."""
    monkeypatch.setattr(
        orchestrator,
        "_fact_extract",
        lambda question, as_of: {
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
    monkeypatch.setattr(
        orchestrator,
        "retrieve_postgres",
        lambda conn, q, as_of, k=5: retrieve_called.append(1) or [],
    )

    body = orchestrator.answer("q", date(2024, 1, 1), object())

    assert body["interrupted"] is True
    assert "Is this a residential or commercial tenancy?" in (body["interrupt_prompt"] or "")
    assert body["results"] == []
    assert retrieve_called == []  # conditional edge bypassed retrieve entirely
```

---

## Invariants to verify before committing

- **Eligibility gate**: `retrieve_postgres` contains the gate. Interrupt path bypasses retrieval entirely — no data fetched, no gate issue. Normal path is unchanged. ✓
- **Model never writes citations**: `_compose_answer` says "Never modify citations" in the prompt. The actual citation dicts in `all_results` come from `validate_and_render` server-side. ✓
- **PS-7 (server-side validation gate)**: `validate_node` still runs before `answer_composer_node` on the normal path. ✓
- **Interrupt response**: `abstained=False`, `results=[]`, `interrupted=True` — semantically correct (asking for info, not abstaining on legal grounds). ✓
- **Dead code removed**: `_classify_and_decompose` and its test deleted. `import os` removed. ✓

## PS requirements in scope

- **PS-6** (per-claim as-of validation) — unchanged; `validate_node` runs before composer
- **PS-7** (server-side validation gate) — unchanged
- **PS-12** (retrieved text is untrusted) — `_compose_answer` prompt says "Never invent law. Use only what the claims provide."

## Zero-tolerance gates

- `repealed-as-current = 0` — `_compose_answer` only formats already-validated claims
- `not-yet-effective-as-current = 0` — unchanged
- `overruled-as-good-law = 0` — unchanged

## Required checks

```bash
make test    # must show 66 passed, 2 skipped
make lint    # must be clean
```

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions, remaining risks.
