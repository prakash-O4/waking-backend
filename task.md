# Task PH-OBS-B: Full stage-level ingestion tracing

**Engineer:** Pi  
**Branch:** `obs/langfuse-ingestion-stages`  
**Base:** `dev` (commit 7236d61)

---

## Objective

Replace the current terminal-only `_emit_ingestion_span` with a proper
per-document trace that wraps all ingestion stages as timed child spans.
Every document — whether it succeeds or fails — must appear in Langfuse with
one span per pipeline stage, each showing stage name, outcome, and latency_ms.

**File in scope: `app/ingestion/pipeline.py` only.**

---

## What exists now (replace this)

`_emit_ingestion_span(source_id, source_type, stage, outcome)` is called only
at exit points (failure or final success). A successfully ingested law produces
exactly one span (`DUAL_APPROVAL_PAUSE / ingested`). Intermediate stages are
invisible. The function also creates a new `Langfuse` client on every call —
replace with a module-level singleton.

---

## Required design

### 1. Module-level Langfuse client singleton

```python
_lf_client: "Langfuse | None" = None

def _get_lf_client() -> "Langfuse | None":
    from app.config import get_settings
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return None
    global _lf_client
    if _lf_client is None:
        from langfuse import Langfuse
        s = get_settings()
        _lf_client = Langfuse(
            public_key=s.LANGFUSE_PUBLIC_KEY,
            secret_key=s.LANGFUSE_SECRET_KEY,
            host=s.LANGFUSE_HOST,
        )
    return _lf_client
```

### 2. One trace per document

At the START of `ingest_law()` and `ingest_nkp_case()`, open a trace:

```python
lf = _get_lf_client()
trace = lf.trace(
    name="ingestion.law",          # or "ingestion.nkp_case"
    metadata={"source_id": source_id, "source_type": source_type},
) if lf else None
```

### 3. One span per stage, with timing

Helper to emit a completed stage span:

```python
def _span(trace: Any, stage: str, *, outcome: str, latency_ms: int) -> None:
    if trace is None:
        return
    trace.span(
        name=f"stage.{stage}",
        metadata={"outcome": outcome, "latency_ms": latency_ms},
    )
```

Pattern inside each stage:

```python
t0 = time.monotonic()
# ... stage work ...
_span(trace, "STAGE_NAME", outcome="passed", latency_ms=int((time.monotonic()-t0)*1000))
```

At every return path (early exit or success), flush before returning:

```python
if lf:
    lf.flush()
```

### 4. Stages to trace

**Laws (`ingest_law`):**

| Stage | outcome values |
|---|---|
| `LOAD` | `skipped` (unchanged hash) or `passed` |
| `VALIDATE` | `rejected` or `passed` |
| `CHUNK` | `rejected` or `passed` |
| `EXTRACT_METADATA` | `failed` (LLM error, caught) or `passed` |
| `EMBED_AND_UPSERT` | `passed` |
| `DUAL_APPROVAL_PAUSE` | `ingested` |

**NKP cases (`ingest_nkp_case`):**

| Stage | outcome values |
|---|---|
| `LOAD` | `skipped` or `passed` |
| `VALIDATE` | `rejected` or `passed` |
| `REDACT_PII` | `quarantined` or `passed` |
| `CHUNK` | `rejected` or `passed` |
| `EXTRACT_METADATA` | `failed` or `passed` |
| `EMBED_AND_UPSERT` | `passed` |
| `DUAL_APPROVAL_PAUSE` | `ingested` |

On early exit, emit only the terminal stage span then flush. Do not emit spans
for stages that were never reached.

---

## PS-14 constraints — unchanged

- `source_id`, stage name, outcome, `latency_ms` in spans: allowed
- `raw_content`, `full_text`, `redacted_text`, any chunk text: never in any span

---

## What does NOT change

- `_langfuse_callback()` and `_emit_answer_trace()` in `gated_orchestrator.py`
- All gate logic, chunkers, parsers, writer, eval harness
- Opt-in: if `LANGFUSE_PUBLIC_KEY` unset, pipeline runs identically

---

## Required checks

```
make test
make lint
```

Return commit hash, checks run/results.

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By`, no AI attribution.
