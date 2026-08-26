# AGENT-7 — Unified Langfuse trace: one tree per query

**Branch:** `agent/stage-7-unified-trace`
**Base:** `dev`
**Engineer:** Pi
**No new ADR needed** — observability change only, no data model or gate logic touched.

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No AI attribution of any kind.

---

## Problem

Every query currently produces 3+ disconnected top-level Langfuse traces:
- `rag.retrieval` — from `retrieve_postgres` (its own `lf.trace()`)
- `rag.answer` — from `_emit_answer_trace` (a fresh `Langfuse()` client, separate trace)
- 1–3 anonymous LLM generation traces — from `LangfuseCallbackHandler`, unlinked

The root traces (`rag.retrieval`, `rag.answer`) both have `endTime: null` because `.end()` is never called on them.

---

## Target state

One trace per query — everything as children:

```
trace: rag.query  [input=query or hash, output=gate_decision+result_count+top_scores]
  ├── generation: fact_extraction     [Gemini — tokens, cost]
  ├── span: retrieval                 [top_vector_score, eligible_count, final_count]
  │   ├── span: stage.eligibility_gate
  │   ├── span: stage.vector_search
  │   ├── span: stage.lexical_search
  │   ├── span: stage.rrf_fusion
  │   ├── span: stage.relevance_gate
  │   └── span: stage.rerank
  ├── span: authority_ranking         [hits_in, hits_out]
  ├── span: cross_ref_resolution      [cross_refs_added]
  ├── generation: reasoning           [Azure gpt-4.1-mini — tokens, cost]
  ├── span: validation                [claims_passed, claims_abstained]
  └── generation: answer_composition  [Gemini — tokens, cost]
```

Root trace `endTime` is always set (non-null).

---

## Acceptance criteria

- `make test` — **67 passed**, 2 skipped (unchanged from AGENT-6)
- `make lint` — clean
- `make eval-gates` — all zero-tolerance gates at 0
- One root `rag.query` trace per `scripts/query.py` run (not two or three)
- Root trace `endTime` non-null
- All LLM generations appear as children of the root trace (not as separate top-level traces)
- Retrieval `stage.*` spans appear under a `retrieval` span, which is a child of the root trace
- `authority_ranking`, `cross_ref_resolution`, `validation` appear as child spans of root trace
- When `LANGFUSE_PUBLIC_KEY` is unset, behaviour is unchanged (no-op throughout)
- No new dependencies, no new files

---

## Scope — exactly these four files

1. `app/retrieval/postgres_retriever.py`
2. `app/retrieval/gated_orchestrator.py`
3. `app/retrieval/query_graph.py`
4. `tests/test_orchestrator.py`

**Do NOT modify** `query_state.py`, `config.py`, `validation_gate.py`, `eligibility_gate.py`, or any other file.

---

## Architecture: trace object via `config["configurable"]`, not `QueryState`

The Langfuse trace object is not JSON-serialisable. Do not add it to `QueryState`. Instead thread it through `config["configurable"]`, which LangGraph passes to every node as a plain Python dict:

```python
# In run_query — passed to _graph.invoke:
config = {"configurable": {"conn": conn, "lf_trace": lf_trace}, "recursion_limit": 10}

# In any node:
lf_trace = config["configurable"].get("lf_trace")   # None when Langfuse is off
```

---

## Part 1 — `postgres_retriever.py`

### 1a. Export `_get_lf_client` → `get_lf_client`

Rename the function (remove underscore). Update the one call site inside the file
(`lf = _get_lf_client()` → `lf = get_lf_client()`).

### 1b. Add `lf_trace: Any = None` to `retrieve_postgres`

```python
def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5, lf_trace: Any = None
) -> list[dict[str, Any]]:
```

### 1c. Replace `lf.trace()` with a child span under `lf_trace`

Delete the current block:
```python
lf = _get_lf_client()
trace = (lf.trace(name="rag.retrieval", metadata={...}) if lf else None)
```

Replace with:
```python
retrieval_span = (
    lf_trace.span(
        name="retrieval",
        metadata={
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "as_of": str(as_of),
            "k": k,
        },
    )
    if lf_trace is not None
    else None
)
```

### 1d. Update all `_end_span(trace, ...)` call sites → `_end_span(retrieval_span, ...)`

There are 6 calls. Rename the variable only — the `_end_span` helper signature is unchanged.

### 1e. Replace `lf.flush()` with `retrieval_span.end()`

Remove all three `if lf: lf.flush()` / `lf.flush()` calls inside `retrieve_postgres`.
Instead end the retrieval span:
- Early exits (`not eligible`, `not candidates`): `if retrieval_span: retrieval_span.end()` before `return []`.
- Normal return: end with summary metadata (see 1f).

The single `_lf.flush()` for the whole pipeline runs in `run_query` after `_graph.invoke()`.

### 1f. End retrieval span with summary at the normal return path

```python
top_vec = max((h.get("vector_score", 0.0) for h in result_hits), default=0.0)
if retrieval_span is not None:
    retrieval_span.end(
        metadata={
            "eligible_count": len(eligible),
            "final_count": len(result_hits),
            "top_vector_score": round(top_vec, 4),
        }
    )
return result_hits
```

---

## Part 2 — `gated_orchestrator.py`

### 2a. `_langfuse_callback(trace_id: str | None = None)`

```python
def _langfuse_callback(trace_id: str | None = None) -> list[Any]:
    settings = get_settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return []
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    except ImportError:
        return []
    kwargs: dict[str, Any] = {
        "public_key": settings.LANGFUSE_PUBLIC_KEY,
        "secret_key": settings.LANGFUSE_SECRET_KEY,
        "host": settings.LANGFUSE_HOST,
    }
    if trace_id:
        kwargs["trace_id"] = trace_id
    return [LangfuseCallbackHandler(**kwargs)]
```

### 2b. `_structured_claims` — add `lf_trace: Any = None`, pass trace_id to callback

```python
def _structured_claims(
    facts: Any,
    issue_queries: list[dict[str, Any]],
    ranked_hits: list[dict[str, Any]],
    lf_trace: Any = None,
) -> dict[str, Any] | None:
```

Inside, replace `callbacks = _langfuse_callback()` with:
```python
trace_id = lf_trace.id if lf_trace is not None else None
callbacks = _langfuse_callback(trace_id)
```

### 2c. `_compose_answer` — add `lf_trace: Any = None`

```python
def _compose_answer(
    facts: Any,
    missing_facts: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    conflict_hits: list[dict[str, Any]],
    session_as_of: date,
    lf_trace: Any = None,
) -> dict[str, Any] | None:
```

Inside, replace `callbacks = _langfuse_callback()` with:
```python
trace_id = lf_trace.id if lf_trace is not None else None
callbacks = _langfuse_callback(trace_id)
```

### 2d. `_fact_extract` — add `lf_trace: Any = None`

```python
def _fact_extract(
    question: str, session_as_of: date, lf_trace: Any = None
) -> dict[str, Any]:
```

Inside, replace `callbacks = _langfuse_callback()` with:
```python
trace_id = lf_trace.id if lf_trace is not None else None
callbacks = _langfuse_callback(trace_id)
```

### 2e. Remove `_emit_answer_trace`. Rewrite `_emit_answer_trace_from_state`

**Delete** `_emit_answer_trace(metadata)` entirely.

**Replace** `_emit_answer_trace_from_state` with:

```python
def _emit_answer_trace_from_state(
    lf_trace: Any,
    raw_query: str,
    session_as_of: date,
    query_type: str,
    all_results: list[dict[str, Any]],
    all_hits: list[dict[str, Any]],
    wall_clock_start: float,
) -> None:
    """Update and end the root Langfuse trace with final pipeline metadata."""
    if lf_trace is None:
        return
    s = get_settings()
    claims_passed = sum(1 for r in all_results if not r.get("abstained"))
    claims_abstained = sum(1 for r in all_results if r.get("abstained"))
    top_chunk_scores = sorted(
        [hit.get("vector_score") or hit.get("score", 0.0) for hit in all_hits],
        reverse=True,
    )[:5]
    output: dict[str, Any] = {
        "gate_decision": "abstained" if not all_results else "answered",
        "result_count": len(all_results),
        "top_chunk_scores": top_chunk_scores,
        "latency_ms": _elapsed_ms(wall_clock_start),
        "validation_claims_passed": claims_passed,
        "validation_claims_abstained": claims_abstained,
    }
    if s.LANGFUSE_LOG_CONTENT:
        output["query"] = raw_query
        for r in all_results:
            if not r.get("abstained"):
                output["answer_summary"] = (
                    r.get("plain_language") or r.get("claim", "")[:200]
                )
                break
    try:
        lf_trace.update(output=output)
        lf_trace.end()
    except Exception:
        pass
```

`lf_trace.end()` is called here (inside `answer_composer_node`). `run_query` only flushes — it does **not** call `lf_trace.end()` again.

---

## Part 3 — `query_graph.py`

### 3a. Import `get_lf_client` from postgres_retriever

Add at the top of imports (after existing `import app.retrieval.gated_orchestrator as _orch`):

```python
import hashlib as _hashlib

from app.retrieval.postgres_retriever import get_lf_client as _get_lf_client
```

### 3b. Rewrite `run_query`

```python
def run_query(question: str, session_as_of: date, conn: Any) -> dict[str, Any]:
    # ── Unified root trace ────────────────────────────────────────────────────
    _lf = _get_lf_client()
    lf_trace = None
    if _lf is not None:
        try:
            s = _orch.get_settings()
            trace_input = (
                question
                if s.LANGFUSE_LOG_CONTENT
                else _hashlib.sha256(question.encode()).hexdigest()[:16]
            )
            lf_trace = _lf.trace(
                name="rag.query",
                input=trace_input,
                metadata={"as_of": session_as_of.isoformat()},
            )
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────────────────

    initial: QueryState = {
        "raw_query": question,
        "session_as_of": session_as_of,
        "subqueries": [],
        "all_hits": [],
        "all_results": [],
        "query_type": "simple",
        "wall_clock_start": _orch.time.monotonic(),
        "facts": None,
        "missing_facts": [],
        "issue_queries": [],
        "interrupted": False,
        "interrupt_prompt": None,
        "_pending_results": [],
        "_response": {},
    }
    result = _graph.invoke(
        initial,
        config={
            "configurable": {"conn": conn, "lf_trace": lf_trace},
            "recursion_limit": 10,
        },
    )

    # ── Flush everything in one shot after all nodes complete ─────────────────
    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────────────────

    return cast(dict[str, Any], result["_response"])
```

### 3c. `fact_extractor_node` — pass `lf_trace`

```python
def fact_extractor_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    lf_trace = config["configurable"].get("lf_trace")
    result = _orch._fact_extract(state["raw_query"], state["session_as_of"], lf_trace=lf_trace)
    # ... rest of function body unchanged ...
```

### 3d. `retrieve_node` — pass `lf_trace` to `retrieve_postgres`

```python
def retrieve_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    all_hits: list[dict[str, Any]] = []

    issue_queries = state["issue_queries"] or [
        {"query": state["raw_query"], "as_of": state["session_as_of"], "work_type_hint": None}
    ]
    for idx, iq in enumerate(issue_queries):
        if _orch._wall_clock_expired(state["wall_clock_start"]):
            break
        hits = _orch.retrieve_postgres(conn, iq["query"], iq["as_of"], k=5, lf_trace=lf_trace)
        for h in hits:
            all_hits.append({**h, "_issue_idx": idx})

    return {"all_hits": all_hits, "_pending_results": []}
```

### 3e. `authority_ranker_node` — add span

```python
def authority_ranker_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    hits_in = len(state["all_hits"])
    ranked = _orch._authority_rank_hits(state["all_hits"], conn)
    if lf_trace is not None:
        try:
            sp = lf_trace.span(name="authority_ranking")
            sp.end(metadata={"hits_in": hits_in, "hits_out": len(ranked)})
        except Exception:
            pass
    return {"all_hits": ranked}
```

### 3f. `cross_ref_resolver_node` — add span

```python
def cross_ref_resolver_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    lf_trace = config["configurable"].get("lf_trace")
    additional = _orch._resolve_cross_refs(
        state["all_hits"], state["session_as_of"], conn
    )
    if lf_trace is not None:
        try:
            sp = lf_trace.span(name="cross_ref_resolution")
            sp.end(metadata={"cross_refs_added": len(additional)})
        except Exception:
            pass
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}
```

### 3g. `reasoner_node` — pass `lf_trace` to `_structured_claims`

```python
parsed = _orch._structured_claims(facts, [iq], issue_hits, lf_trace=lf_trace)
```

Add `lf_trace = config["configurable"].get("lf_trace")` at the top of the function.

### 3h. `validate_node` — add span

After building `all_results`, add:
```python
if lf_trace is not None:
    try:
        sp = lf_trace.span(name="validation")
        sp.end(metadata={
            "claims_passed": sum(1 for r in all_results if not r.get("abstained")),
            "claims_abstained": sum(1 for r in all_results if r.get("abstained")),
        })
    except Exception:
        pass
```

Add `lf_trace = config["configurable"].get("lf_trace")` at the top.

### 3i. `answer_composer_node` — new body

**Order matters:** call `_compose_answer` first (so its generation fires before the root trace ends), then call `_emit_answer_trace_from_state` (which ends the trace).

```python
def answer_composer_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    lf_trace = config["configurable"].get("lf_trace")
    session_as_of = state["session_as_of"]
    query_type = state["query_type"]

    if state.get("interrupted"):
        if lf_trace is not None:
            try:
                lf_trace.update(output={"interrupted": True, "result_count": 0})
                lf_trace.end()
            except Exception:
                pass
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

    # Compose first so its Langfuse generation fires before the root trace ends
    composed = _orch._compose_answer(
        state["facts"],
        state["missing_facts"],
        all_results,
        conflict_hits,
        session_as_of,
        lf_trace=lf_trace,
    )

    # End the root trace (sets output + endTime)
    _orch._emit_answer_trace_from_state(
        lf_trace,
        raw_query,
        session_as_of,
        query_type,
        all_results,
        all_hits,
        state["wall_clock_start"],
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

---

## Part 4 — `tests/test_orchestrator.py`

### 4a. Replace `test_emit_trace_uses_vector_score`

The function now takes `lf_trace` as its first arg (an object, not a hook into `_emit_answer_trace`).

```python
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
```

### 4b. No other test changes needed

- Tests that mock `_fact_extract`, `_structured_claims`, `_compose_answer` mock them at the module level — they don't call them directly, so the new `lf_trace=None` default parameter is transparent.
- `test_compose_answer_no_key_returns_none` calls `orchestrator._compose_answer(None, [], [...], [], date(2024,1,1))` — `lf_trace` defaults to `None`, still works.
- `test_required_missing_fact_returns_interrupted_response` calls `orchestrator.answer(...)` which goes through `run_query`. `_get_lf_client()` returns `None` in test (no `LANGFUSE_PUBLIC_KEY` set), so `lf_trace=None` throughout. Works unchanged.

---

## Invariants to verify before committing

- **PS-14** — `LANGFUSE_LOG_CONTENT` still gates raw query/answer. Default `False`.
- **PS-6/7/12** — All Langfuse calls in nodes are in `try/except Exception: pass` blocks. A Langfuse failure cannot break a query.
- **No gate path change** — eligibility gate, validation gate, retrieval, generation are unchanged. Spans are side-effect observers only.
- **Zero-tolerance gates** — no change to `_compose_answer` data path; fence stripping from AGENT-6 preserved.

## Required checks

```bash
make test    # 67 passed, 2 skipped
make lint    # clean
```

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions, remaining risks.
