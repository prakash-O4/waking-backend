# Task PH-OBS-A: Langfuse RAG tracing integration

**Engineer:** Pi  
**Branch:** `obs/langfuse-tracing`  
**Base:** `dev` (commit 292059b)

---

## Objective

Wire Langfuse as the traceability layer for the Wakil-G query and ingestion
pipelines. Langfuse runs self-hosted; connection is configured via env vars.
**PS-14 is the primary constraint** — traces store IDs/hashes, never raw
legal text.

---

## Acceptance criteria

1. `langfuse>=2.0` added to `requirements.txt`; no version conflict with
   existing deps (langchain 1.3.14, langchain-openai 1.4.1, openai).
2. Three new settings in `app/config.py`:
   ```
   LANGFUSE_PUBLIC_KEY: str = ""
   LANGFUSE_SECRET_KEY: str = ""
   LANGFUSE_HOST: str = "http://localhost:3000"
   ```
3. `LangfuseCallbackHandler` wired into the LangChain LLM calls inside
   `app/retrieval/gated_orchestrator.py`. When `LANGFUSE_PUBLIC_KEY` is set,
   calls appear as spans in Langfuse automatically.
4. `answer()` in `gated_orchestrator.py` emits a top-level Langfuse trace
   containing **only**:
   - `query_hash` — SHA-256 of the question (NOT the raw question string)
   - `as_of` — the date string
   - `query_type` — `simple | complex | extractive`
   - `latency_ms`
   - `retrieved_uris` — list of `component_uri` strings from hits (NOT `text_ne`)
   - `gate_decision` — `answered | abstained`
   - `result_count`
5. `IngestionPipeline` in `app/ingestion/pipeline.py` emits per-document spans
   containing **only**: `source_id`, `source_type`, stage name, outcome
   (`ingested | skipped | rejected | quarantined`). Never `raw_content`,
   `full_text`, or `redacted_text`.
6. Langfuse is **optional**: if `LANGFUSE_PUBLIC_KEY` is empty/unset, no
   Langfuse client initializes and the pipeline runs identically to before —
   no import error, no crash, no changed behaviour.
7. `make test` green. `make lint` green.
8. Zero-tolerance eval gates untouched.

---

## PS-14 hard constraints — never violate

| What | Rule |
|---|---|
| Query text | SHA-256 hash only — never the raw string in any span |
| Retrieved statutory text | `component_uri` only — never `text_ne` |
| LLM prompts / completions | Do NOT add to span metadata; LangChain callbacks handle these — do not re-add |
| `raw_content`, `full_text`, `redacted_text` | Never in any span, ever |

system-design.md §11 governs. Redaction is at the instrumentation layer, not
the Langfuse server.

---

## Files in scope

- `requirements.txt`
- `app/config.py`
- `app/retrieval/gated_orchestrator.py`
- `app/ingestion/pipeline.py`

## Files explicitly out of scope

- Any gate logic (`eligibility_gate.py`, `validation_gate.py`, `gates.py`)
- Chunkers, parsers, writer, calendar
- Eval harness or eval metrics
- `app/main.py`
- Langfuse server setup (ops, not this task)
- LangSmith wiring in `app/main.py` (separate cleanup task)

---

## Required checks

```
make test
make lint
```

Run both. Return results.

---

## Commit authorship

Every commit on this branch:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By`, no AI attribution in commit messages or files.
