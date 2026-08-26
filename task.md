# AGENT-8 — Production-grade ingestion observability

**Branch:** `agent/obs-ingestion-spans`
**Base:** `dev`
**Engineer:** Pi
**No new ADR needed** — observability only; no gate logic, no data model touched.

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No AI attribution of any kind.

---

## Problem

The ingestion pipeline is a black box. Three concrete failures:

1. **`endTime: null` on every Langfuse span** — `_span()` in `pipeline.py` calls `trace.span(...)` but never stores or ends the returned object. Langfuse cannot compute duration. Every timing value in the UI is blank.

2. **No inputs or outputs on spans** — spans carry only `{"outcome": "passed", "latency_ms": N}`. Zero context: no chunk count, no LLM call count, no extracted metadata. Impossible to diagnose why a record took 60 seconds.

3. **Terminal is silent inside a record** — operator sees `[N/total] source_id …` then silence for up to 60 seconds. No way to know which stage is running.

---

## Target state

### Terminal output per record
```
[324/400] 93654191-... …
  LOAD          20ms
  VALIDATE       0ms
  CHUNK          8ms   68 chunks
  EXTRACT_META  43.2s  4 batches · 3 LLM calls
  EMBED+UPSERT   2.1s  68 chunks
  ✓ ingested    total=47.2s
```
- Two-space indent. Stage name left-padded to 12 chars. Latency immediately after. Short annotation.
- Use `print(..., flush=True)` so it streams immediately.
- Latency: `<1s` → `ms` (e.g. `8ms`); `>=1s` → `s` with one decimal (e.g. `43.2s`).

### Langfuse spans — required inputs/outputs

| Stage | `input` | `output` |
|---|---|---|
| LOAD | `{"source_id": ..., "source_type": ..., "content_len": N}` | `{"outcome": "passed/skipped", "is_amendment": bool}` |
| VALIDATE | `{"content_len": N}` | `{"outcome": "passed/rejected", "has_dafa_anchor": bool}` |
| CHUNK | `{"content_len": N}` | `{"outcome": "passed/rejected", "chunk_count": N}` |
| EXTRACT_METADATA | `{"chunk_count": N, "batch_count": N, "llm_calls": N}` | `{"outcome": "passed/failed", "summary_extracted": bool, "keywords_extracted": N}` |
| EMBED_AND_UPSERT | `{"chunk_count": N}` | `{"outcome": "passed", "document_id": "..."}` |
| DUAL_APPROVAL_PAUSE | `{"document_id": "..."}` | `{"outcome": "ingested"}` |

**All spans must have non-null `endTime`.** Use the create-then-end pattern:
```python
span = trace.span(name="stage.CHUNK", input={"content_len": len(content)})
chunks = self._laws_chunker.chunk_text(content)
span.end(output={"outcome": "passed", "chunk_count": len(chunks)})
```

---

## Implementation

### 1. Replace `_span()` in `pipeline.py` with `_begin_span` / `_end_span`

```python
def _begin_span(trace: Any, stage: str, input: dict) -> Any:
    if trace is None:
        return None
    try:
        return trace.span(name=f"stage.{stage}", input=input)
    except Exception:
        return None

def _end_span(span: Any, output: dict) -> None:
    if span is None:
        return
    try:
        span.end(output=output)
    except Exception:
        pass
```

Delete the old `_span()` entirely.

### 2. Add `_fmt_latency(seconds: float) -> str`

```python
def _fmt_latency(s: float) -> str:
    return f"{int(s * 1000)}ms" if s < 1.0 else f"{s:.1f}s"
```

### 3. Refactor each stage in `ingest_law` and `ingest_nkp_case`

For each stage:
- `span = _begin_span(trace, "STAGE", input={...})` before the work
- do the work, capture results into local variables
- `_end_span(span, output={...})`
- `print(f"  {'STAGE':<12} {_fmt_latency(elapsed)}  {annotation}", flush=True)`

Print the total at the end:
```python
print(f"  {'total':<12} {_fmt_latency(time.monotonic() - record_start)}", flush=True)
```
Track `record_start = time.monotonic()` at the top of `ingest_law` / `ingest_nkp_case`.

For the skipped-record early exit, still print:
```
  LOAD          3ms   skipped (unchanged)
```

### 4. LLM call count from `metadata_enricher`

Change `enrich_law_chunks` and `enrich_nkp_chunks` to return `(metadata, llm_call_count)` tuple.

`llm_call_count` = 1 (summary/case-level call) + `len(batches)` (chunk batch calls).

`_parallel_chunk_metadata` should return `(metadata, batch_count)` so callers know both.

`pipeline.py` unpacks the tuples and uses `llm_call_count` in the EXTRACT_METADATA span and stdout annotation.

---

## Allowed files

- `app/ingestion/pipeline.py` — primary target
- `app/ingestion/metadata_enricher.py` — add `llm_call_count` return values
- `tests/test_ingestion_pipeline.py` — update for changed signatures; add tests below

**Do NOT touch** any other file.

---

## New tests required in `test_ingestion_pipeline.py`

1. **`test_langfuse_span_end_called`** — mock a trace object on `ingest_law` (use `enable_llm=False`); assert `.end()` is called on every stage span (LOAD, VALIDATE, CHUNK, EMBED_AND_UPSERT, DUAL_APPROVAL_PAUSE).

2. **`test_enrich_law_llm_call_count`** — mock the LLM; call `enrich_law_chunks` with N chunks; assert `llm_call_count == 1 + ceil(N / 20)`.

---

## Forbidden

- Do not change any gate predicate, eligibility logic, chunk logic, or retrieval path
- Do not modify any file outside the allowed list
- Do not add new dependencies
- Do not change ingestion behaviour — only add observability side-effects

---

## Required checks

```bash
make test    # all existing tests must still pass
make lint    # clean
```

## Zero-tolerance gates in scope
None — pure observability change.

---

## Return to Claude

- Commit hash
- Changed files list
- `make test` full output
- `make lint` full output
- Assumptions and remaining risks
