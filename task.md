# AGENT-6 — Observability Fix: meaningful Langfuse traces + compose JSON parse

**Branch:** `agent/stage-6-observability-fix`
**Base:** `dev`
**Engineer:** Pi
**ADR:** `docs/adr-001-multi-agent-query-architecture.md` — no new ADR needed (all fixes to existing mechanisms)

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No "Generated with Claude" line. No AI attribution of any kind.

---

## Objective

Four observable problems with the current Langfuse traces:

1. **All spans have `endTime: null`** — `.end()` is never called on span objects returned by `trace.span()`.
2. **`top_chunk_scores` are RRF scores (`~0.016`) not vector cosine scores** — misleading metric in the answer trace.
3. **No LLM generation events / no token cost** — the `LangfuseCallbackHandler` for the Azure reasoner call is never flushed; Gemini calls have no Langfuse callbacks at all.
4. **`_compose_answer` always returns `None`** — Gemini wraps JSON in markdown fences (` ```json … ``` `); `json.loads` fails and the `except Exception: return None` silently discards it, so the ADR Node 7 answer format is never delivered.

Plus a PS-14-compliant content logging flag: raw query and plain-language answer must be opt-in only, off by default.

---

## Acceptance criteria

- `make test` — **66 tests passing**, 2 skipped (same count as AGENT-5)
- `make lint` — clean
- `make eval-gates` — zero-tolerance gates unchanged: `repealed-as-current=0`, `not-yet-effective-as-current=0`, `overruled-as-good-law=0`
- All retrieval spans have a non-null `endTime` in Langfuse
- `top_chunk_scores` in the `rag.answer` trace are cosine similarity floats (range 0–1), not RRF scores
- When `LANGFUSE_LOG_CONTENT=true` is set, the `rag.answer` trace includes `query` (raw question text) and `answer_summary` (the `plain_language` field from the composer, or first claim text if composer returns None)
- When `LANGFUSE_LOG_CONTENT=false` (default), only the hash is logged — PS-14 maintained
- The Azure LLM callback handler is explicitly flushed after `_structured_claims` returns
- Both Gemini calls (`_fact_extract`, `_compose_answer`) pass `_langfuse_callback()` so their generations appear in Langfuse with token counts
- `_compose_answer` successfully parses Gemini responses that are wrapped in markdown code fences

---

## Scope — exactly these four files

1. `app/config.py`
2. `app/retrieval/postgres_retriever.py`
3. `app/retrieval/gated_orchestrator.py`
4. `tests/test_orchestrator.py` (update/add tests as needed to cover new behavior)

**Do NOT modify** `query_graph.py`, `query_state.py`, `validation_gate.py`, or any other file.

---

## Part 1 — `config.py`

### 1a. Add `LANGFUSE_LOG_CONTENT` field

In the `Settings` class, after the `LANGFUSE_HOST` line, add:

```python
LANGFUSE_LOG_CONTENT: bool = False
```

No other changes to `config.py`.

---

## Part 2 — `postgres_retriever.py`

### 2a. Store span objects and call `.end()`

Currently `_span()` calls `trace.span()` but discards the return value. Spans need `.end()` called to record `endTime`.

Replace the `_span` helper and update every call site:

```python
def _end_span(trace: Any, stage: str, **metadata: Any) -> None:
    """Create a span and immediately end it so Langfuse records endTime."""
    if trace is not None:
        span = trace.span(name=f"stage.{stage}", metadata=metadata)
        span.end()
```

Rename all calls from `_span(trace, ...)` to `_end_span(trace, ...)` throughout the file. There are 6 call sites.

### 2b. Carry vector cosine scores through to the returned hits

Currently `_hit()` sets `"score": float(score)` where `score` is the RRF score. The vector cosine score is computed during vector search but not stored.

Change `_hit()` to also accept an optional `vector_score`:

```python
def _hit(row: tuple[Any, ...], score: float, vector_score: float = 0.0) -> dict[str, Any]:
    chunk_id, text, text_hash, act_name, case_id, chunk_type, section_number = row[:7]
    source_id = row[7] if len(row) == 8 else ""
    return {
        "component_uri": str(chunk_id),
        "text_ne": text,
        "text_hash": text_hash,
        "score": float(score),           # RRF score — used for ranking
        "vector_score": float(vector_score),  # cosine similarity — used for observability
        "chunk_type": chunk_type,
        "section_number": section_number or "",
        "document_source_id": str(source_id) if source_id else "",
        "work_title_ne": act_name or case_id or "",
    }
```

During vector search, pass the cosine similarity as `vector_score`. The RRF pipeline builds a `rows` dict of `chunk_id → row`. For the final returned hits, look up the cosine score from the vector results. Specifically:

- After `vector_search()` returns rows, build `vector_scores: dict[str, float]` mapping `chunk_id → vec_score` (the last column of the vector search query).
- In the final `_hit()` call (line ~290), pass `vector_score=vector_scores.get(h["component_uri"], 0.0)`.

The SQL already returns `vec_score` as the last column — it just isn't stored anywhere. Use it.

---

## Part 3 — `gated_orchestrator.py`

### 3a. Fix `_compose_answer` — strip markdown fences before `json.loads`

Gemini wraps responses in ` ```json … ``` `. Add stripping before parse:

```python
raw = str(resp.content).strip()
# Strip optional markdown code fences
if raw.startswith("```"):
    raw = raw.split("```", 2)[1]          # drop opening fence line
    if raw.startswith("json"):
        raw = raw[4:]                      # drop "json" language tag
    raw = raw.rsplit("```", 1)[0].strip() # drop closing fence
return cast(dict[str, Any], json.loads(raw))
```

Replace the current one-liner `return cast(dict[str, Any], json.loads(str(resp.content).strip()))` with this.

### 3b. Add Langfuse callbacks to both Gemini calls

In `_fact_extract`, the Gemini `llm.invoke()` call currently has no callbacks. Add them:

```python
resp = llm.invoke(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    config={"callbacks": _langfuse_callback()},
)
```

Do the same in `_compose_answer`:

```python
resp = llm.invoke(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    config={"callbacks": _langfuse_callback()},
)
```

### 3c. Flush callback handler after `_structured_claims` Azure LLM call

After the `resp = llm.invoke(...)` line in `_structured_claims`, flush the callback:

```python
callbacks = _langfuse_callback()
resp = llm.invoke(
    [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    config={"callbacks": callbacks},
)
if callbacks:
    try:
        callbacks[0].langfuse.flush()
    except Exception:
        pass
return cast(dict[str, Any], json.loads(str(resp.content).strip()))
```

Apply the same flush pattern in `_fact_extract` and `_compose_answer` after their `llm.invoke()` calls.

### 3d. Fix `_emit_answer_trace_from_state` — use cosine scores, add content when flag is on

Replace the current `_emit_answer_trace_from_state` implementation:

```python
def _emit_answer_trace_from_state(
    raw_query: str,
    session_as_of: date,
    query_type: str,
    all_results: list[dict[str, Any]],
    all_hits: list[dict[str, Any]],
    wall_clock_start: float,
) -> None:
    """Extracted from answer() for graph node use."""
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return
    try:
        __import__("langfuse")
    except ImportError:
        return
    s = get_settings()
    claims_passed = sum(1 for r in all_results if not r.get("abstained"))
    claims_abstained = sum(1 for r in all_results if r.get("abstained"))
    # Use vector cosine scores (meaningful 0–1 range), fall back to RRF score
    top_chunk_scores = sorted(
        [hit.get("vector_score") or hit.get("score", 0.0) for hit in all_hits],
        reverse=True,
    )[:5]
    metadata: dict[str, Any] = {
        "query_hash": hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
        "as_of": session_as_of.isoformat(),
        "query_type": query_type,
        "latency_ms": _elapsed_ms(wall_clock_start),
        "gate_decision": "abstained" if not all_results else "answered",
        "result_count": len(all_results),
        "top_chunk_scores": top_chunk_scores,
        "validation_claims_passed": claims_passed,
        "validation_claims_abstained": claims_abstained,
    }
    if s.LANGFUSE_LOG_CONTENT:
        metadata["query"] = raw_query
        # Include first non-abstained plain_language or first claim text
        for r in all_results:
            if not r.get("abstained"):
                metadata["answer_summary"] = r.get("plain_language") or r.get("claim", "")[:200]
                break
    _emit_answer_trace(metadata)
```

---

## Part 4 — `tests/test_orchestrator.py`

### 4a. Update `test_compose_answer_success` to use markdown-wrapped JSON

Gemini now returns markdown fences in real calls. Update the `FakeResp.content` to use a markdown-wrapped response to verify the stripping works:

```python
class FakeResp:
    content = "```json\n" + _json.dumps({...}) + "\n```"
```

The existing assertion `assert result is not None` will verify the fence-stripping works.

### 4b. Add test for `_emit_answer_trace_from_state` content logging

Add one test verifying that `vector_score` is preferred over `score` in `top_chunk_scores`:

```python
def test_emit_trace_uses_vector_score(monkeypatch: Any) -> None:
    """top_chunk_scores uses vector_score when present, not RRF score."""
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(orchestrator, "_emit_answer_trace", captured.append)
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: SimpleNamespace(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_LOG_CONTENT=False),
    )
    __import__("sys").modules.setdefault("langfuse", object())  # satisfy import check

    hits = [
        {"score": 0.016, "vector_score": 0.71},
        {"score": 0.015, "vector_score": 0.65},
    ]
    orchestrator._emit_answer_trace_from_state("q", date(2024, 1, 1), "simple", [], hits, 0.0)

    assert captured
    scores = captured[0]["top_chunk_scores"]
    assert scores[0] == pytest.approx(0.71)
    assert scores[1] == pytest.approx(0.65)
```

---

## Invariants to verify before committing

- **PS-6** — `validate_node` runs before `answer_composer_node` on normal path; unchanged.
- **PS-7** — Server-side validation gate unchanged.
- **PS-12** — Retrieved text still wrapped untrusted; `_compose_answer` prompt unchanged.
- **PS-14** — Raw query only logged when `LANGFUSE_LOG_CONTENT=True`. Default is `False`. This field must never be set in a CI/CD environment without explicit intent.
- **Zero-tolerance gates** — `_compose_answer` only formats already-validated claims; fence-stripping doesn't affect the gate logic.

## PS requirements in scope

- **PS-14** (privacy / observability) — `LANGFUSE_LOG_CONTENT` flag is the enforcement mechanism.

## Zero-tolerance gates in scope

None of the three gates are touched by this change.

## Required checks

```bash
make test    # must show 66 passed (or 67 if test_emit_trace_uses_vector_score is new), 2 skipped
make lint    # must be clean
```

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions, remaining risks.
