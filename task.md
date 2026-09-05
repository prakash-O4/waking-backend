# AGENT-37: minimal dev console for /ask

## Objective

Prakash needs a dev-oriented way to test `/ask` without curl/Postman —
pose a question, see the full raw response (a "console") to debug what
the pipeline actually returned, including abstentions and degraded-mode
flags. Not a product feature, not hifi — a single-file internal tool.

Doubles as the practical way to generate real, varied test traffic for
roadmap item 4 (AGENT-36's claim-support labeling) now that the product
is pre-launch and Langfuse only has placeholder `"q"` smoke-test traces
(see `.agent/PROGRESS.md`, 2026-09-04 entry) — posing real questions
through this console still hits `/ask` normally, so they land in Langfuse
exactly like any other call and become fetchable via
`scripts/label_eval_candidates.py --fetch` afterward.

## Grounding (already done, don't re-derive)

- `/ask` (`app/main.py:43-64`) is the only endpoint that matters here.
  Takes `POST {question: string, as_of?: "YYYY-MM-DD"}`, requires
  `Authorization: Bearer <supabase-jwt>` (checked via
  `SupabaseHelper.get_user_id`/`check_daily_quota`,
  `app/utils/helpers.py`) — a missing/invalid token 401s, quota-exceeded
  404s. Returns `orchestrator_answer(...)` directly, which is
  `query_graph.py::run_query()`'s `_response` dict: `as_of`, `query_type`,
  `abstained`, `results: [{claim, evidence_id, abstained, citation}]`,
  `degraded_mode: [...]`, and on interrupt: `interrupted: true`,
  `interrupt_prompt`. This is already a rich debug payload — no backend
  change needed to get a useful console.
- CORS is already wide open (`app/main.py:31-36`,
  `allow_origins=["*"]`) — a fully static page can call `/ask` directly
  from any origin (`file://`, a plain local static server, whatever),
  no backend change needed for that either.
- **Decided with Prakash (2026-09-04)**: auth is handled by a plain
  paste-your-own-JWT input field in the console, sent as-is on each
  request, kept only in the browser's own `localStorage`. Do **not**
  add any auth bypass, dev-only endpoint, or change to
  `app/utils/helpers.py`/`app/main.py`'s auth logic — that was
  explicitly rejected as a bigger, riskier change than this task needs.
- `scripts/` is this repo's existing home for human-invoked,
  never-served dev tooling (`label_eval_candidates.py`, `migrate.py`,
  `query.py`, etc.) — reuse it rather than inventing a new top-level
  directory (Ponytail: no new file category).

## What to build

One new file: `scripts/dev_console.html`. Single self-contained static
HTML file — inline `<style>` and `<script>`, vanilla JS only. No
framework, no npm, no build step, no new dependency of any kind. Opened
directly in a browser (`file://...scripts/dev_console.html`) or served
via `python3 -m http.server` from anywhere — it is never wired into the
FastAPI app, never imported by any Python module, never referenced from
`main.py`.

Required UI/behavior:

1. **Two persisted settings fields** (persist in `localStorage`, not
   hardcoded, not committed with any value):
   - API base URL, default `http://localhost:8000`.
   - Authorization token (raw JWT, no "Bearer " prefix required from the
     user — the script prepends it). Starts empty. Never bake in a
     default token.
2. **Request form**: a question textarea, and an optional `as_of` date
   input (blank = omit the field entirely so the API's own
   `date.today()` default applies — do not default it to today
   yourself, that's the API's job).
3. **Submit** → `fetch(baseUrl + "/ask", {method: "POST", headers:
   {"Content-Type": "application/json", "Authorization": "Bearer " +
   token}, body: JSON.stringify({question, ...(as_of ? {as_of} : {})})
   })`.
4. **A running console/log pane** below the form: each submission
   appends a new timestamped entry showing the exact request payload
   sent, the HTTP status code, and the full response body
   pretty-printed (`JSON.stringify(body, null, 2)` in a `<pre>`, or
   equivalent). New entries must **append**, not replace, the previous
   ones — the whole point is a scrollable session log you can look back
   through. Show non-2xx responses and network/fetch errors (e.g. CORS
   failure, connection refused) in the same log pane, clearly marked as
   errors — do not fail silently or only show a blank/console.error.
5. **A "Clear console" button** that empties the log pane only (does
   not touch the two persisted settings fields, does not touch any
   file).
6. Keep it visually plain — no design polish required, just usable:
   labeled inputs, a submit button, monospace log output. Basic
   responsive layout is enough (doesn't need to work on mobile).

## Explicitly forbidden

- Any change to `app/main.py`, `app/retrieval/*`, `app/utils/helpers.py`,
  or any other backend file. This task adds exactly one new static file
  and touches nothing else.
- Any new backend route, static-file mount, or wiring of this file into
  the FastAPI app.
- Any auth bypass, mock token, or dev-only credential shortcut.
- Any JS framework, bundler, package.json, or node dependency — plain
  `<script>` tag, vanilla JS, one file.
- Replicating `--show`'s pre-gate debug view (raw retrieval hits,
  claims before validation) — `/ask` doesn't expose those, and building
  a new endpoint to expose internal pipeline state is a separate,
  bigger ask with its own Ponytail/PS-check pass. Out of scope here.
- Hardcoding any real token, API key, or non-localhost URL into the
  committed file.

## Required checks

No `make test`/`make lint` target applies to a static HTML file (the
Makefile's fixed lint list is Python-only) — run `make test && make
lint` anyway just to confirm nothing else was accidentally touched or
broken (should trivially pass, zero Python files changed).

Manual verification (describe what you did in the completion report,
since there's no automated test for a static page):
- Open the file in a browser, set the base URL to a locally-running
  `uvicorn app.main:app` instance, paste a real token, submit a
  question — confirm the response renders in the log pane.
- Submit with a deliberately bad/empty token — confirm the 401 renders
  visibly in the log pane, not a silent failure.
- Submit twice — confirm both entries remain visible (append, not
  replace).
- Reload the page — confirm the base URL and token fields persisted,
  and the log pane is empty (log itself is not required to persist).

No `make eval-gates` relevance — this touches no gate/temporal/
ingestion/precedent code; it's a static client hitting an already-
existing, already-reviewed endpoint, unmodified.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line. Use:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.

## Branch

`agent/dev-console`, off `dev`.
