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

## What's broken today

1. **`endTime: null` on every Langfuse span.** `_span()` calls `trace.span()` and discards the returned object. Langfuse never sees `.end()`, so every span has null duration.

2. **Spans carry no inputs or outputs.** Only `{"outcome": "passed", "latency_ms": N}` in metadata. No chunk count, no extracted data, nothing to understand a slow record.

3. **LLM calls are invisible.** The 3–5 Azure gpt-4.1-mini calls inside `metadata_enricher.py` are fire-and-forget. No Langfuse generation is created. You cannot see prompt, response, token counts, or cost for any of them.

4. **Embedding call is invisible.** The Azure `text-embedding-3-large` call in `pgvector_indexer.py` has no trace. Token count is lost from the API response every time.

5. **Terminal is silent inside a record.** Operator sees `[N/total] source_id …` then silence for up to 60s.

---

## Target: what the Langfuse trace must look like

Production LLM pipelines use three Langfuse observation types:
- **Trace** — root container per document
- **Span** — non-LLM steps (DB ops, chunking, upsert)
- **Generation** — every LLM call (has `model`, `input`, `output`, `usage_details` with tokens, auto-computed cost)

Required tree per law document:

```
trace: ingestion.law
  input:  { source_id, source_type, content_len }
  output: { outcome, chunk_count, total_llm_calls, total_input_tokens, total_output_tokens }

  ├── span: LOAD
  │     input:  { source_id, source_type, content_len }
  │     output: { outcome: "passed"|"skipped", is_amendment: bool }

  ├── span: VALIDATE
  │     input:  { content_len }
  │     output: { outcome: "passed"|"rejected", has_dafa_anchor: bool }

  ├── span: CHUNK
  │     input:  { content_len }
  │     output: { outcome: "passed"|"rejected", chunk_count: N }

  ├── span: EXTRACT_METADATA
  │     input:  { chunk_count: N, batch_count: N, llm_calls: N }
  │     output: { outcome: "passed"|"failed", summary_extracted: bool,
  │               keywords_extracted: N, input_tokens: N, output_tokens: N }
  │
  │     ├── generation: act_summary           ← one LLM call for the act summary
  │     │     model:   gpt-4.1-mini (from settings.AZURE_OPENAI_LLM_DEPLOYMENT)
  │     │     input:   prompt text (or "[redacted N chars]" when LANGFUSE_LOG_CONTENT=False)
  │     │     output:  response text (or "[redacted N chars]")
  │     │     usage_details: { input: N, output: N, total: N }   ← from API response
  │     │
  │     ├── generation: chunk_keywords_batch_0   ← one per batch (ceil(chunks/20))
  │     │     model / input / output / usage_details  (same pattern)
  │     ├── generation: chunk_keywords_batch_1
  │     └── … (all batches, including parallel ones)

  ├── span: EMBED_AND_UPSERT
  │     input:  { chunk_count: N }
  │     output: { outcome: "passed", document_id: "...", total_embedding_tokens: N }
  │
  │     └── generation: embedding
  │           model:         text-embedding-3-large (from settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
  │           input:         { chunk_count: N, batch_count: N }
  │           output:        { vectors_produced: N }
  │           usage_details: { input: N, total: N }   ← from response.usage.total_tokens

  └── span: DUAL_APPROVAL_PAUSE
        input:  { document_id: "..." }
        output: { outcome: "ingested", status: "pending" }
```

Same tree structure applies to `ingest_nkp_case`. For NKP there are also:
- `REDACT_PII` span (no LLM; just input/output counts)
- `enrich_nkp_chunks` has a `nkp_case_metadata` generation instead of `act_summary`

---

## Target: terminal output per record

```
[324/400] 93654191-79ec-5abc-902d-32cdf1605f1c …
  LOAD          20ms    new document · 45,230 chars
  VALIDATE       0ms    दफा anchor found
  CHUNK          8ms    → 68 chunks
  EXTRACT_META  43.2s   5 LLM calls · 12,450 in + 3,210 out tok
  EMBED+UPSERT   2.1s   68 chunks · 34,200 embed tok
  ✓ ingested    total=47.2s
```

Rules:
- `print(..., flush=True)` — streams immediately, not buffered
- Stage name left-padded to 12 chars
- Latency: `<1s` → `Nms`; `≥1s` → `N.Ns`
- Skipped records still print the LOAD line: `  LOAD  3ms  skipped (unchanged)`
- Rejected records print every stage up to rejection, then `  ✗ rejected`

---

## Implementation — exactly what to change

### File 1: `app/ingestion/pipeline.py`

#### 1a. Replace `_span()` with `_begin_span()` + `_end_span()`

Delete the current `_span()` function. Add:

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

#### 1b. Add `_fmt_latency()`

```python
def _fmt_latency(s: float) -> str:
    return f"{int(s * 1000)}ms" if s < 1.0 else f"{s:.1f}s"
```

#### 1c. Add `record_start = time.monotonic()` at the top of `ingest_law` and `ingest_nkp_case`

Track total elapsed for the final summary print.

#### 1d. Rewrite each stage block to: begin span → do work → end span → print

Example for CHUNK:
```python
t0 = time.monotonic()
span = _begin_span(trace, "CHUNK", input={"content_len": len(content)})
chunks = self._laws_chunker.chunk_text(content)
elapsed = time.monotonic() - t0
if not chunks:
    _end_span(span, output={"outcome": "rejected", "chunk_count": 0})
    print(f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → 0 chunks", flush=True)
    # ... rejection handling ...
    return None
_end_span(span, output={"outcome": "passed", "chunk_count": len(chunks)})
print(f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → {len(chunks)} chunks", flush=True)
```

Apply this pattern to LOAD, VALIDATE, CHUNK, EXTRACT_METADATA, EMBED_AND_UPSERT, DUAL_APPROVAL_PAUSE (and REDACT_PII for NKP).

#### 1e. Pass `lf_parent=span` to enricher (EXTRACT_METADATA span → enricher creates generations under it)

```python
span = _begin_span(trace, "EXTRACT_METADATA", input={
    "chunk_count": len(chunks),
    "batch_count": math.ceil(len(chunks) / metadata_enricher.CHUNK_BATCH_SIZE),
    "llm_calls": 1 + math.ceil(len(chunks) / metadata_enricher.CHUNK_BATCH_SIZE),
})
t0 = time.monotonic()
metadata, llm_calls, in_tok, out_tok = metadata_enricher.enrich_law_chunks(record, chunks, lf_parent=span)
elapsed = time.monotonic() - t0
_end_span(span, output={
    "outcome": metadata_outcome,
    "llm_calls": llm_calls,
    "input_tokens": in_tok,
    "output_tokens": out_tok,
    "summary_extracted": bool(summary),
    "keywords_extracted": sum(1 for m in metadata if m.get("keywords")),
})
print(f"  {'EXTRACT_META':<12} {_fmt_latency(elapsed)}  {llm_calls} LLM calls · {in_tok:,} in + {out_tok:,} out tok", flush=True)
```

Add `import math` at the top of `pipeline.py`.

#### 1f. Pass `lf_parent=embed_span` to `embed_chunks` (EMBED_AND_UPSERT span → indexer creates generation under it)

```python
embed_span = _begin_span(trace, "EMBED_AND_UPSERT", input={"chunk_count": len(chunks)})
t0 = time.monotonic()
embeddings, embed_tok = self._embed(
    [c.embed_text for c in chunks], lf_parent=embed_span
)
document_id = self._indexer.upsert_document(document, chunks, embeddings)
self._conn.commit()
elapsed = time.monotonic() - t0
_end_span(embed_span, output={
    "outcome": "passed",
    "document_id": document_id,
    "total_embedding_tokens": embed_tok,
})
print(f"  {'EMBED+UPSERT':<12} {_fmt_latency(elapsed)}  {len(chunks)} chunks · {embed_tok:,} embed tok", flush=True)
```

#### 1g. Update `_embed()` to return `(embeddings, total_tokens)`

```python
def _embed(self, texts: list[str], lf_parent: Any = None) -> tuple[list[list[float]], int]:
    embeddings, total_tokens = self._indexer.embed_chunks(texts, lf_parent=lf_parent)
    return embeddings, total_tokens
```

#### 1h. End trace with summary output

After DUAL_APPROVAL_PAUSE and before returning:
```python
if trace is not None:
    try:
        trace.update(output={
            "outcome": "ingested",
            "chunk_count": len(chunks),
            "total_llm_calls": llm_calls,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "total_embedding_tokens": embed_tok,
        })
        trace.end()
    except Exception:
        pass
print(f"  {'total':<12} {_fmt_latency(time.monotonic() - record_start)}", flush=True)
```

For skipped and rejected paths, also call `trace.end()` before returning.

---

### File 2: `app/ingestion/metadata_enricher.py`

#### 2a. Change `_call_llm()` signature

```python
def _call_llm(
    prompt: str,
    lf_parent: Any = None,
    generation_name: str = "llm_call",
) -> tuple[str, dict]:
    """Returns (content, usage) where usage = {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}."""
```

Inside:
1. Before invoke: create `gen = lf_parent.generation(name=generation_name, model=deployment, input=gen_input)` if `lf_parent` is not None. `gen_input` = prompt if `LANGFUSE_LOG_CONTENT` else `f"[redacted {len(prompt)} chars]"`.
2. After invoke: extract `usage = response.response_metadata.get("token_usage", {})` from the LangChain response object. Call `gen.end(output=gen_output, usage_details={"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0), "total": usage.get("total_tokens", 0)})`. `gen_output` = content if `LANGFUSE_LOG_CONTENT` else `f"[redacted {len(content)} chars]"`.
3. On exception: `gen.end(level="ERROR", status_message=str(exc))` then re-raise.
4. Return `(content, usage)` — a tuple.

The deployment name comes from `get_settings().AZURE_OPENAI_LLM_DEPLOYMENT`.

#### 2b. Update `_process` inside `_parallel_chunk_metadata`

```python
def _process(batch: list[Any], batch_idx: int) -> tuple[list[dict], dict]:
    prompt = _chunk_metadata_prompt(batch, intro)
    raw, usage = _call_llm(prompt, lf_parent=lf_parent, generation_name=f"chunk_keywords_batch_{batch_idx}")
    return _apply_chunk_metadata(batch, raw), usage
```

`lf_parent` is now a parameter of `_parallel_chunk_metadata`.

#### 2c. Change `_parallel_chunk_metadata` signature and return type

```python
def _parallel_chunk_metadata(
    chunks: list[Any], intro: str, lf_parent: Any = None
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Returns (metadata, batch_count, total_input_tokens, total_output_tokens)."""
```

Accumulate token counts across all futures:
```python
total_in = total_out = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LLM) as pool:
    futures = {pool.submit(_process, batch, idx): batch for idx, batch in enumerate(batches)}
    for future in as_completed(futures):
        try:
            batch_meta, usage = future.result()
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            for entry in batch_meta:
                metadata[entry["chunk_index"]].update(entry)
        except Exception as exc:
            logger.warning(f"parallel chunk-metadata batch failed: {exc}")

return metadata, len(batches), total_in, total_out
```

#### 2d. Change `enrich_law_chunks` signature and return type

```python
def enrich_law_chunks(
    act_record: dict[str, Any],
    chunks: list[LawChunk],
    lf_parent: Any = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Returns (metadata, llm_call_count, total_input_tokens, total_output_tokens)."""
```

- Summary LLM call: `raw, usage = _call_llm(prompt, lf_parent=lf_parent, generation_name="act_summary")`
- Chunk batch calls: `metadata, batch_count, chunk_in, chunk_out = _parallel_chunk_metadata(chunks, intro, lf_parent=lf_parent)`
- Accumulate: `total_in = usage.get("prompt_tokens", 0) + chunk_in`, etc.
- Return: `metadata_with_summary, 1 + batch_count, total_in, total_out`

#### 2e. Change `enrich_nkp_chunks` signature and return type

Same pattern. The case-level LLM call uses `generation_name="nkp_case_metadata"`. Returns `(metadata, 1 + batch_count, total_in, total_out)`.

---

### File 3: `app/ingestion/pgvector_indexer.py`

#### 3a. Change `embed_chunks` signature and return type

```python
def embed_chunks(
    self,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    lf_parent: Any = None,
) -> tuple[list[list[float]], int]:
    """Returns (embeddings, total_tokens)."""
```

#### 3b. Create Langfuse generation before embedding loop, end it after

```python
gen = None
if lf_parent is not None:
    try:
        gen = lf_parent.generation(
            name="embedding",
            model=deployment,
            input={
                "chunk_count": len(texts),
                "batch_count": math.ceil(len(texts) / batch_size),
            },
        )
    except Exception:
        pass

embeddings: list[list[float]] = []
total_tokens = 0
for start in range(0, len(texts), batch_size):
    batch = texts[start: start + batch_size]
    response = client.embeddings.create(model=deployment, input=batch, dimensions=dims)
    embeddings.extend([d.embedding for d in response.data])
    total_tokens += response.usage.total_tokens  # Azure OpenAI always returns this

if gen is not None:
    try:
        gen.end(
            output={"vectors_produced": len(embeddings)},
            usage_details={"input": total_tokens, "total": total_tokens},
        )
    except Exception:
        pass

return embeddings, total_tokens
```

Add `import math` at the top of the file.

---

### File 4: `tests/test_ingestion_pipeline.py`

#### 4a. Update all call sites of `metadata_enricher.enrich_law_chunks` and `enrich_nkp_chunks`

They now return 4-tuples. Any mock or direct call must unpack `(metadata, llm_calls, in_tok, out_tok)`.

#### 4b. Update all call sites of `pgvector_indexer.embed_chunks` mocks

They now return `(embeddings_list, token_count_int)`.

#### 4c. Add `test_langfuse_span_end_called`

```python
def test_langfuse_span_end_called(monkeypatch, ...):
    """Every stage span must have .end() called — endTime must not be null."""
    ended_spans = []

    class FakeSpan:
        def __init__(self, name):
            self.name = name
        def end(self, output=None, **kwargs):
            ended_spans.append(self.name)

    class FakeTrace:
        def span(self, name, input=None):
            return FakeSpan(name)
        def generation(self, **kwargs):
            return FakeSpan(kwargs.get("name", "gen"))
        def update(self, **kwargs): pass
        def end(self): pass

    # Mock enricher to return 4-tuple, indexer embed to return ([], 0)
    # Run ingest_law with enable_llm=False and a valid minimal record
    # Assert "stage.LOAD", "stage.VALIDATE", "stage.CHUNK",
    #        "stage.EMBED_AND_UPSERT", "stage.DUAL_APPROVAL_PAUSE"
    # all appear in ended_spans
```

#### 4d. Add `test_enrich_law_llm_call_count`

```python
def test_enrich_law_llm_call_count(monkeypatch):
    """llm_call_count == 1 + ceil(chunk_count / CHUNK_BATCH_SIZE)."""
    import math
    # Mock _call_llm to return ("[]", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    # Create N=45 fake chunks
    # Call enrich_law_chunks
    # Assert llm_call_count == 1 + math.ceil(45 / CHUNK_BATCH_SIZE)
    # Assert total_input_tokens == 10 * (1 + ceil(45/20)) — each call returns 10 prompt tokens
```

---

## PS-14 compliance

`LANGFUSE_LOG_CONTENT` (default `False`) must gate all raw content:
- When `False`: `gen.input` = `f"[redacted {len(prompt)} chars]"`, `gen.output` = `f"[redacted {len(response)} chars]"`
- When `True`: actual prompt and response text
- Token counts are **never** gated — they contain no content, only counts
- Embedding `input` dict (chunk_count, batch_count) contains no text — always logged

This maintains the existing PS-14 guarantee: no raw statutory text in traces by default.

---

## Allowed files

- `app/ingestion/pipeline.py`
- `app/ingestion/metadata_enricher.py`
- `app/ingestion/pgvector_indexer.py`
- `tests/test_ingestion_pipeline.py`

**Do NOT touch** any other file. No gate logic, no chunker, no retriever, no query path.

---

## Required checks

```bash
make test    # all existing tests must still pass
make lint    # clean
```

## Zero-tolerance gates in scope
None — pure observability.

---

## Return to Claude

- Commit hash
- Changed files
- `make test` full output
- `make lint` full output
- Assumptions and remaining risks
