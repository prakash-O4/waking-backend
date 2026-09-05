# AGENT-38: /ask/stream — live phase visibility for the dev console

## Objective

Prakash wants the dev console (`scripts/dev_console.html`, AGENT-37) to
show *which pipeline stage is currently running* while a question is
being answered, not just the final blob. Add a new streaming endpoint
that reports stage-by-stage progress in real time, and wire the console
to consume it.

**This is a safety-critical spec, not a normal feature add. Read the
whole "Non-negotiable safety contract" section before writing any code.**

## Grounding (already done, don't re-derive)

- The pipeline is a LangGraph `StateGraph` built in
  `app/retrieval/query_graph.py:491-518` (`build_graph()` →
  module-level `_graph = build_graph()`). Nodes, in the order they can
  execute: `fact_extractor → retrieve → authority_ranker →
  co_retrieve_parent_resolver → cross_ref_resolver →
  enabling_power_resolver → reasoner → validate → answer_composer`.
  `fact_extractor` has a conditional edge straight to `answer_composer`
  when the state is `interrupted` — so `answer_composer` is *always*
  the last node to execute, on every path, interrupted or not.
- `run_query()` (`query_graph.py:524-573`) is the existing synchronous
  entry point: builds a Langfuse trace, builds the `initial: QueryState`
  dict, calls `_graph.invoke(initial, config={"configurable": {"conn":
  conn, "lf_trace": lf_trace}, "recursion_limit": 10})`, flushes
  Langfuse, returns `result["_response"]`. `answer_composer_node`
  (ends at `query_graph.py:482-485`) is what actually produces
  `{"_response": composed}` — `composed` is already the fully
  validated, gate-passed response object that `/ask` returns today.
  Nothing upstream of `answer_composer` produces `_response`.
- `app/main.py`'s `/ask` calls `orchestrator_answer` = `answer()` in
  `app/retrieval/gated_orchestrator.py:600-603`, which lazily imports
  and calls `query_graph.run_query()` (lazy import there to dodge a
  circular import — `query_graph.py` imports `gated_orchestrator as
  _orch` at module scope). Follow that same lazy-import pattern for the
  new streaming path, don't fight it.
- LangGraph's compiled graph supports `.stream(initial, config=...,
  stream_mode="updates")`, which yields one `{node_name: state_delta}`
  dict per node as it finishes — the *same* execution as `.invoke()`,
  just observed incrementally. No new dependency needed (LangGraph is
  already a dependency; `StreamingResponse` is already in
  `fastapi.responses`, same module `JSONResponse` already comes from).
- `tests/test_ask_pipeline.py` is the existing convention for testing
  `/ask` via `TestClient` with a `FakeSupabaseHelper` and `fake_connect`
  — follow the same pattern for the new endpoint's tests.

## Non-negotiable safety contract

Core Invariant 4 in `system-design.md` §2: *"A server-side validation
gate is the production gate... No answer bypasses it."* Invariant 7:
*"Abstention is server-owned."* Streaming a node's raw output before
`validate`/`answer_composer` has run would show the user
unvalidated/unabstained content — that's a second, ungated path to an
"answer," which is exactly what those invariants forbid. This is not a
style preference, it's the reason this task exists as its own reviewed
task instead of a five-line change.

**Rule, mechanically enforced in code, not just by convention:**

- For every node *except* `answer_composer`, the streaming code may
  read the **dict key only** (the node name) from each
  `{node_name: state_delta}` step. It must never index into, log,
  serialize, or forward `state_delta` (the value) for any of these
  nodes. Not even for debugging. If you find yourself writing
  `step[stage]["something"]` where `stage != "answer_composer"`, stop —
  that's the bug this task exists to prevent.
- For the `answer_composer` step only, read exactly one field:
  `step["answer_composer"]["_response"]`. That's the same object
  `run_query()`/`/ask` already returns today — it has already passed
  the validation gate by the time this node produces it. Do not read
  or forward any other key from that step either.
- The event shape sent over the wire, for every non-final stage:
  `{"stage": "<node_name>", "status": "done", "latency_ms": <int>}`.
  Nothing else. For the final event:
  `{"stage": "final", "status": "done", "response": <the _response object>}`.
- No "running" event before a stage starts. LangGraph's conditional
  routing means you don't know which node runs next until it's already
  finished (the interrupt branch skips straight to `answer_composer`),
  so emitting an honest "running: X" would require hardcoding the
  graph's edge structure into the streaming code — a second copy of
  routing logic that can drift from the real graph. Don't do that.
  "Done" events arriving one at a time as they complete already gives
  live visibility; that's the whole feature.
- If anything raises inside the generator (DB error, node exception,
  anything), catch it and yield exactly one
  `{"stage": "error", "status": "error", "detail": "<short message, no
  stack trace, no raw content>"}` event, then stop — don't let the
  connection just die silently, but don't leak exception internals
  either (avoid an OWASP-style info-disclosure leak: no tracebacks, no
  SQL, no file paths in `detail`).

## What to build

### 1. `app/retrieval/query_graph.py`

- Factor the trace-setup block (`run_query()` lines ~525-541) into a
  small private helper, e.g. `_start_trace(question, session_as_of)`,
  returning `lf_trace` (or `None`). `run_query()` calls it; the new
  function below calls it too. Pure extraction, no behavior change —
  `run_query()`'s existing behavior and existing tests must be
  unaffected.
- Factor the `initial: QueryState = {...}` dict (lines ~543-558) into a
  small private helper, e.g. `_initial_state(question, session_as_of)`.
  Same rule: pure extraction, `run_query()` unaffected.
- Add `stream_query(question: str, session_as_of: date, conn: Any) ->
  Iterator[dict[str, Any]]`: builds `lf_trace`/`initial` via the two
  helpers above, iterates `_graph.stream(initial, config={...same
  config shape as run_query...}, stream_mode="updates")`, and for each
  step yields the sanitized event per the contract above (tracking
  elapsed time between yields for `latency_ms`, e.g. via
  `_orch.time.monotonic()` the same way `run_query`'s `initial` dict
  already does). Flush Langfuse (same `_lf.flush()` pattern as
  `run_query`) once the loop completes, before/at the final yield.

### 2. `app/retrieval/gated_orchestrator.py`

- Add a thin wrapper mirroring `answer()` (lines 600-603):
  `stream_answer(question, session_as_of, conn) -> Iterator[dict[str,
  Any]]` that lazily imports `query_graph` (same reason as `answer()`)
  and yields from `query_graph.stream_query(...)`. Keep `answer()`
  itself untouched.

### 3. `app/main.py`

- Extract the existing auth+quota block from `ask_question()` (currently
  inline: `SupabaseHelper()` → `get_user_id` → `check_daily_quota` →
  `HTTPException(404, ...)`) into a small helper, e.g. `_authorize
  (authorization: Optional[str]) -> str` returning `user_id`, raising
  the exact same exceptions with the exact same status codes/messages
  as today. `ask_question()` calls this helper — pure refactor, **every
  existing test in `tests/test_ask_pipeline.py` must keep passing
  unmodified** (that's your regression check that the refactor changed
  nothing observable).
- Add `POST /ask/stream`, same `AskRequest` body as `/ask`:
  ```python
  @app.post("/ask/stream")
  async def ask_question_stream(
      req: AskRequest, authorization: Optional[str] = Header(default=None)
  ) -> StreamingResponse:
      user_id = _authorize(authorization)
      as_of = req.as_of or date.today()

      def event_gen():
          try:
              with connect() as conn:
                  for event in stream_answer(req.question, as_of, conn):
                      yield f"data: {json.dumps(event)}\n\n"
          except psycopg2.OperationalError:
              yield f"data: {json.dumps({'stage': 'error', 'status': 'error', 'detail': 'database unavailable'})}\n\n"

      return StreamingResponse(event_gen(), media_type="text/event-stream")
  ```
  (Illustrative — adjust naming/imports to match the file's actual
  style, but keep the event shapes and the try/except-around-the-whole-
  generator structure.) Import `stream_answer` from
  `app.retrieval.gated_orchestrator` alongside the existing
  `orchestrator_answer` import. Import `StreamingResponse` from
  `fastapi.responses` (same module as the existing `JSONResponse`
  import) and stdlib `json`.
- `/ask` itself: **zero behavior change**, only the internal
  auth/quota extraction above.

### 4. `scripts/dev_console.html`

- Change the Submit handler to call `/ask/stream` instead of `/ask`.
  `EventSource` doesn't support POST bodies or custom headers, so use
  `fetch` with a manually-read `ReadableStream`
  (`res.body.getReader()`, decode chunks with `TextDecoder`, split on
  `\n\n` to find complete SSE `data: ...` lines, `JSON.parse` each).
- On request start, create one log entry (reuse the existing
  `addEntry`/append-only pane) showing the request payload, then as
  each stage event arrives, append a line to *that same entry* (not a
  new top-level entry per stage) — e.g. `retrieve: done (842ms)`, one
  line per stage, appended live as they arrive.
- When the `"final"` event arrives, render its `response` field
  pretty-printed exactly like the old `/ask` response rendering did
  (reuse the existing `JSON.stringify(..., null, 2)` /
  `<pre>` approach).
- If the initial HTTP response itself is non-2xx (e.g. 401 bad token,
  404 quota) before any streaming data arrives, or a `{"stage":
  "error", ...}` event arrives, render it clearly as an error in the
  same entry — same "never fail silently" rule as before.
- No change to Login/email/password/base-URL behavior from the AGENT-37
  amendment — this task only touches the Submit path.
- Still one static file, vanilla JS, no framework/dependency.

## Explicitly forbidden

- Any raw chunk text, any claim/quote before validation, any
  citation not produced by the `answer_composer`/`validate` path, ever
  appearing in a stream event. This is the one thing this review will
  fail on immediately if violated — see "Non-negotiable safety
  contract" above.
- Hardcoding the graph's node order/edges into the streaming code (in
  `query_graph.py`, `gated_orchestrator.py`, or `main.py`) to fabricate
  "running" events. Only "done" events, only from what LangGraph
  itself actually reports.
- Any change to `/ask`'s observable behavior (status codes, response
  shape, error handling) — the refactor must be invisible to existing
  callers and existing tests.
- Any new pip/npm dependency. `StreamingResponse` and LangGraph's
  `.stream()` are already available in this stack.
- Any change to `eligibility_gate.py`, `validation_gate.py`, or any
  node function's logic — this task only changes how already-computed,
  already-gated results are *observed*, never what gets computed.

## Required checks

- `make test && make lint` — must stay fully green, including every
  existing test in `tests/test_ask_pipeline.py` unmodified and passing.
- New tests (extend `tests/test_ask_pipeline.py` or add
  `tests/test_query_graph_stream.py`, your call) must include, at
  minimum:
  - A unit test on `stream_query`/`stream_answer` with a fake/monkey-
    patched graph (or fake node functions) where a non-final node's
    state delta contains an obviously-sensitive key (e.g.
    `{"all_hits": ["raw law text"]}`) — assert the yielded event for
    that stage is exactly `{"stage": ..., "status": "done",
    "latency_ms": ...}` and does **not** contain `all_hits` or any
    other key from the delta. This is the test that enforces the whole
    point of this task — do not skip it or make it toothless.
  - A test that the final yielded event's `"response"` equals what
    `run_query()` would return for equivalent input (same fixture
    style as `test_ask_pipeline.py`'s `fake_answer`).
  - `/ask/stream` endpoint tests mirroring
    `test_ask_delegates_to_orchestrator` and
    `test_ask_quota_still_blocks` — same auth/quota behavior, now
    against the streaming endpoint.
- Manual verification (describe in the completion report, no browser
  available in your environment — same caveat as AGENT-37): open the
  console against a locally running backend, submit a question, confirm
  stage lines append one at a time (or at least arrive as a distinct
  ordered list matching graph order) ending in the same final answer
  `/ask` would have given for that question.

## Commit authorship

Every commit authored `Prakash Basnet <basnetprakash090@gmail.com>` — no
AI attribution, no `Co-Authored-By: Claude` trailer.

## Branch

`agent/ask-stream`, off `dev`.
