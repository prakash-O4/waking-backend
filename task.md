# task.md — AGENT-44: stop the synchronous Langfuse flush from blocking `/ask`/`/ask/stream`

## How this was found

While debugging AGENT-43, Prakash pasted an OTel export error:
`Failed to export span batch code: None, reason:
HTTPSConnectionPool(host='jp.cloud.langfuse.com', port=443): Read timed
out. (read timeout=4.9999730587005615)`. Investigation confirmed this
is a real, live latency bug, independent of RAG correctness — already
flagged in `.agent/PROGRESS.md`'s AGENT-43 entry as queued but
unscoped.

## Root cause (confirmed via git history + Langfuse/OTel SDK source)

`_flush_langfuse()` (`app/retrieval/query_graph.py:594-600`) calls
`_lf.flush()` synchronously and inline in the request path — in a
`finally` block in `run_query()` (`/ask`, line 639) and right before
the final SSE event in `stream_query()` (`/ask/stream`, lines 663-664,
674-675, 678-679). `Langfuse.flush()` triggers OTEL's
`BatchSpanProcessor.force_flush()`, which does a **real, blocking HTTP
POST** to Langfuse Cloud on the calling thread — confirmed directly in
the installed SDK source
(`opentelemetry/sdk/_shared_internal/__init__.py`, comment: `# Blocking
call to export.`). The client has no configured timeout
(`app/retrieval/postgres_retriever.py:56-60` constructs `Langfuse(...)`
with no `timeout=`), so the SDK's default of **5 seconds** applies —
exactly matching the observed `read timeout=4.99...` error. Whenever
Langfuse Cloud is slow, every `/ask` call and the final answer of every
`/ask/stream` call eats up to 5 seconds of latency that has nothing to
do with computing the actual answer.

**Confirmed via `git log -p`**: these synchronous flush calls
(`ec899a1` "Unify Langfuse query tracing", `7e47500` "Add streaming ask
endpoint") were added with no stated reason — no commit message
explanation, no code comment justifying why the flush needs to be
synchronous/inline.

**Confirmed via the installed SDK source**
(`langfuse/_client/span_processor.py`,
`opentelemetry/sdk/trace/export/__init__.py`): the `Langfuse` client
already constructs an OTEL `BatchSpanProcessor` with its own
**background worker thread** that auto-exports spans every 5 seconds
by default (`_DEFAULT_SCHEDULE_DELAY_MILLIS = 5000`), completely
independent of any explicit `.flush()` call and completely off the
request thread. `app/retrieval/postgres_retriever.py:46-61` constructs
this client as a module-level object — the background thread starts
once and lives for the lifetime of the process.

**Confirmed via `Dockerfile`**: this app runs as a persistent `uvicorn`
server (`CMD ["uvicorn", "app.main:app", ...]`), not a short-lived/
serverless process — so the background batch-export thread is
guaranteed to keep running and eventually upload every span on its own
schedule, with no risk of losing traces on process exit.

**Confirmed no code depends on immediate post-response trace
availability**: `scripts/label_eval_candidates.py --fetch` (the only
consumer of uploaded traces besides the Langfuse UI itself) is a
separate, manually-invoked CLI command run well after the fact, not
something that races the `/ask` response. No test asserts flush timing
on the query path.

**Conclusion**: the explicit synchronous flush is pure, unjustified
latency cost. The background batch processor Langfuse's own SDK
already runs does the same job for free, off the request thread. The
fix is deletion, not concurrency — do not add threading, thread pools,
or any new async mechanism; that would be building something the SDK
already provides.

## Objective

Remove the synchronous `_flush_langfuse()` calls from the `/ask` and
`/ask/stream` request paths in `app/retrieval/query_graph.py`. Let
Langfuse's own background `BatchSpanProcessor` upload spans on its
normal ~5-second schedule, off the request thread.

## Exact change required

In `app/retrieval/query_graph.py`:

1. Delete the `_flush_langfuse()` function entirely (currently lines
   594-600).

2. In `run_query()` (currently lines 622-641): remove the `finally:
   _flush_langfuse()` block. Resulting function:
   ```python
   def run_query(
       question: str, session_as_of: date, conn: Any, user_id: str | None = None
   ) -> dict[str, Any]:
       with _propagation_scope(user_id):
           lf_trace = _start_trace(question, session_as_of)
           try:
               result = _graph.invoke(
                   _initial_state(question, session_as_of),
                   config={
                       "configurable": {"conn": conn, "lf_trace": lf_trace},
                       "recursion_limit": 10,
                   },
               )
           except Exception as e:
               _trace_error(lf_trace, "query_pipeline", e, fatal=True)
               raise

       return cast(dict[str, Any], result["_response"])
   ```

3. In `stream_query()` (currently lines 644-681): remove the `flushed`
   bookkeeping variable and all four `_flush_langfuse()` call sites (the
   inline call + `flushed = True` right before the `"final"` yield, and
   both `if not flushed: _flush_langfuse()` guards). Resulting function:
   ```python
   def stream_query(
       question: str, session_as_of: date, conn: Any, user_id: str | None = None
   ) -> Iterator[dict[str, Any]]:
       with _propagation_scope(user_id):
           lf_trace = _start_trace(question, session_as_of)
           last = _orch.time.monotonic()
           try:
               for step in _graph.stream(
                   _initial_state(question, session_as_of),
                   config={
                       "configurable": {"conn": conn, "lf_trace": lf_trace},
                       "recursion_limit": 10,
                   },
                   stream_mode="updates",
               ):
                   for stage in step:
                       if stage == "answer_composer":
                           response = step["answer_composer"]["_response"]
                           yield {"stage": "final", "status": "done", "response": response}
                       else:
                           now = _orch.time.monotonic()
                           yield {
                               "stage": stage,
                               "status": "done",
                               "latency_ms": int((now - last) * 1000),
                           }
                           last = now
           except Exception as e:
               _trace_error(lf_trace, "stream_query_pipeline", e, fatal=True)
               yield {"stage": "error", "status": "error", "detail": "query failed"}
   ```

That's the entire change. Confirmed via grep: `_flush_langfuse` and
`flushed` are referenced nowhere else in the repo (no other app code,
no test) — this is a fully self-contained deletion.

## Acceptance criteria

- `_flush_langfuse()` and all five of its call sites are gone, exactly
  as shown above.
- `_get_lf_client`, `_start_trace`, `_trace_error`, and every other
  Langfuse helper in this file are untouched — only the explicit flush
  mechanism is removed.
- No behavior change to what gets traced or how — spans are still
  created and still get uploaded, just on Langfuse's own background
  schedule instead of being forced inline.
- `make test`/`make lint` — **use `.venv/bin/python`, not bare
  `python3`** (see AGENT-40/42/43's entries in `.agent/PROGRESS.md` for
  why).
- Manual smoke test: run a real `/ask` and a real `/ask/stream` request
  against the live server, confirm the response returns without any
  added flush-related delay, and confirm (via the Langfuse UI or
  `scripts/label_eval_candidates.py --fetch` a few seconds later) that
  the trace still shows up normally.

## Explicitly forbidden

- Do **not** add threading, a thread pool, `asyncio.create_task`, or
  any new concurrency mechanism. The fix is deletion — Langfuse's SDK
  already runs its own background export thread; adding another
  mechanism on top would be redundant complexity for something that
  already works for free.
- Do **not** change the Langfuse client construction in
  `app/retrieval/postgres_retriever.py` (no new timeout config, no new
  `flush_at`/`flush_interval` tuning) — out of scope, the default
  5-second background schedule is what we're relying on.
- Do **not** touch `app/ingestion/pipeline.py`'s `_flush()` calls — that
  one is legitimately justified (short-lived CLI ingestion script, real
  risk of losing a trace on early process exit) and is a completely
  separate code path from this bug.
- Do **not** touch any gate, temporal, ingestion, or precedent logic —
  this task is pure observability plumbing, no PS-* requirement is
  affected.

## Governing references

None of the Core Invariants or PS-1…PS-18 apply here — this is
observability-only. No gate logic, no citation logic, no temporal
logic touched.

## Required checks

- `.venv/bin/python -m pytest tests/`
- `.venv/bin/python -m ruff check` / `ruff format --check` / `mypy
  --strict` on the exact Makefile file list (substitute
  `.venv/bin/python` for `python3`)
- Manual live smoke test as described above

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, Pi, or any AI
attribution. Enforce via:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`
No `Co-Authored-By` trailers, no "Generated with" lines.
