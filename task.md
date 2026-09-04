# task.md — AGENT-34: eval-labeling tooling

## Objective

Post-AGENT-33 roadmap item 2. Build a script/workflow that turns real
`/ask` query traffic into golden eval entries with minimal friction, with
Prakash's own review as the labeling authority. This step builds the
*tooling* only — it does not add new golden data itself, and it does not
touch any production/serve path. Golden-set expansion responsibly needs
Prakash's own legal verification; this tool makes that verification fast.

This directly unblocks roadmap item 4 (deciding whether the claim-support
verifier needs to move beyond quote-substring matching), which explicitly
requires "a labeled (quote, claim) eval slice — built using step 2's
tooling" before any entailment/NLI model is even considered.

## Grounding (confirmed this session, don't re-derive)

- No query-logging table exists in this backend. But every real `/ask`
  call creates a Langfuse trace named `"rag.query"`
  (`app/retrieval/query_graph.py::run_query`, line ~524-540), and `.env`
  has `LANGFUSE_LOG_CONTENT=true` with live cloud credentials
  (`LANGFUSE_HOST=https://jp.cloud.langfuse.com`) — so real question text
  + `as_of` (in `trace.metadata["as_of"]`) is genuinely being captured
  today. This is the traffic source (confirmed with Prakash directly,
  AskUserQuestion, over the alternative of an unused Supabase `chat`
  table that this backend never reads or writes — leave that alone,
  out of scope).
- `app/retrieval/postgres_retriever.py::get_lf_client()` already
  constructs and caches the Langfuse client from settings — **reuse it
  as-is**, do not construct a second client.
- Installed SDK: `langfuse==2.60.10`. `Langfuse.fetch_traces(name=...,
  limit=..., from_timestamp=..., order_by=...)` returns
  `FetchTracesResponse` with `.data: list[TraceWithDetails]`. Each trace
  object has `.id`, `.timestamp`, `.input`, `.metadata` (confirmed via
  `TraceWithDetails.__fields__` this session).
- **Content-availability trap**: when `LANGFUSE_LOG_CONTENT` was `false`
  at trace time, `trace_input` is
  `hashlib.sha256(question.encode()).hexdigest()[:16]` (see
  `query_graph.py::run_query`), not the real question — a 16-char lowercase
  hex string. `--fetch` must detect this heuristically (all-hex, exactly
  16 chars) and mark the candidate `content_available: false`; `--show`
  and `--label` must both refuse to run on such a candidate (`SystemExit`
  with a clear message), since the real question text is unrecoverable.
  This heuristic is a convenience, not the safety boundary — the actual
  safety boundary is Prakash reading the real question text himself at
  `--show`/`--label` time before it ever reaches a golden file.
- **`validate_and_render()` drops the quote.** `app/retrieval/validation_gate.py::validate_and_render`
  takes claims (each `{claim, evidence_id, quote}`) and returns
  `{claim, evidence_id, abstained, citation}` — no `quote` in the output.
  Do **not** modify `validate_and_render` to add it (that's the production
  gate — out of scope, no reason to touch it). Instead call the pipeline
  in three explicit steps, all already-existing, unmodified functions:
  1. `app.retrieval.postgres_retriever.retrieve_postgres(conn, question, as_of)`
     → ranked hits (same function `retrieval_slice.py`/`phase_a_slice.py`
     already use for eval).
  2. `app.retrieval.gated_orchestrator._structured_claims(None, [{"query": question, "as_of": as_of}], hits)`
     → `{"claims": [{"claim", "evidence_id", "quote", "issue", "applicability", "condition"}], "abstain": bool}`.
     This is already imported cross-module by `query_graph.py` as
     `_orch._structured_claims` — importing it from a new script is
     consistent with existing precedent, not a new violation of Python
     privacy convention.
  3. `app.retrieval.validation_gate.validate_and_render(claims["claims"], as_of, conn)`
     → current gate verdict per claim, for reference/comparison only (do
     **not** treat this as ground truth — the whole point of the eval
     slice is to measure where the human label disagrees with this gate).
- Existing golden file shapes (do not conflate — keep the new golden
  files separate, tagged with provenance, rather than silently merging
  into hand-curated `romanized.json`/`phase_a_qa.json`):
  - retrieval-recall: `{query, as_of, expected_uris, reference?, note?}`
    (`app/eval/golden/romanized.json`, `phase_c_romanized.json`).
  - RAGAS QA: `{query, as_of, note}`, no expected answer at all
    (`phase_a_qa.json`).
  - **No existing file has `(quote, claim, supports)` shape** — this task
    creates it.
- CLI convention to mirror: `scripts/review_documents.py` /
  `scripts/review_lifecycle.py` — argparse, flag-driven (not an
  interactive REPL loop), `--by <name>` required on any action that
  records a judgment, mutually exclusive top-level action group.

## Locked design

New file: **`scripts/label_eval_candidates.py`**

State files (new, all under `app/eval/golden/`):
- `_traffic_queue.json` — list of
  `{trace_id, question, as_of, as_of_source: "metadata"|"trace_timestamp_fallback",
  content_available: bool, fetched_at, status: "pending"|"labeled"|"skipped",
  reason?}`. One entry per Langfuse trace ever fetched — `--fetch` must
  dedupe against this file by `trace_id` so re-running `--fetch` doesn't
  requeue already-seen traces.
- `labeled_traffic.json` — retrieval-recall shape, extended:
  `{query, as_of, expected_uris, source: "langfuse:<trace_id>", labeled_by, labeled_at}`.
- `claim_support.json` — new shape:
  `{query, as_of, quote, claim, evidence_id, supports: bool,
  gate_verdict_abstained: bool, source: "langfuse:<trace_id>", labeled_by, labeled_at}`.

Both output files: **append-only** — an existing file's prior entries must
survive every run (read-modify-write the full list, never truncate).

Commands (argparse, mutually exclusive group, matching
`review_documents.py`'s style):

- `--fetch [--since-days N=7] [--limit N=50]`
  Calls `get_lf_client().fetch_traces(name="rag.query", from_timestamp=now-N days, limit=N, order_by="timestamp.desc")`.
  For each trace not already in `_traffic_queue.json` (by `trace_id`):
  detect hash-content per the trap above; extract `as_of` from
  `trace.metadata.get("as_of")`, falling back to
  `trace.timestamp.date().isoformat()` (tag which source was used); append
  as `status: "pending"`. Print count of new candidates queued.

- `--list [--status pending|labeled|skipped]` (default `pending`)
  Print each matching candidate: `trace_id  as_of  question` (truncate long
  questions for display only, store full text).

- `--show <trace_id>`
  Refuse (`SystemExit`, clear message) if `content_available` is false or
  trace not found. Otherwise run the three-step pipeline above **fresh**
  (never cached) against the live DB and print: ranked hits (component_uri,
  act/case identifying fields, tier), then each claim indexed
  (`[0] claim=... evidence_id=... quote=...`), then the current gate's
  verdict per claim from `validate_and_render`. Read-only — writes nothing.

- `--label <trace_id> --by <name> [--uris uri1,uri2,...] [--claim "IDX:supports"] [--claim "IDX:refutes"] [--claim "IDX:skip"]...`
  Re-runs the same three-step pipeline **fresh** (not reused from a prior
  `--show` call — the corpus may have changed in between; this matters
  for the same temporal-correctness reason AGENT-33 exists). Validates
  every `--claim IDX` against the freshly-returned claims list length —
  reject with a clear error if `IDX` is out of range (guards against
  labeling stale indices from an old `--show`). Requires `--by`. If
  `--uris` given (non-empty), appends one entry to `labeled_traffic.json`.
  For every `--claim IDX:supports` or `IDX:refutes` (not `:skip`), appends
  one entry to `claim_support.json` with `supports` set accordingly and
  `gate_verdict_abstained` taken from this fresh run's
  `validate_and_render` output for that claim. At least one of `--uris` or
  `--claim` must be given. Marks the queue entry `status: "labeled"`.

- `--skip <trace_id> --by <name> --reason "<text>"`
  Marks `status: "skipped"`. No golden entries written. For candidates
  that are unusable (garbage query, pure abstain, off-topic, anything
  Prakash doesn't want captured — no automated PII scrubbing is built
  here; `--skip` at human-review time is the safety gate for that).

## Explicitly forbidden

- Do not modify `validation_gate.py`, `query_graph.py`,
  `gated_orchestrator.py`, `postgres_retriever.py`, `main.py`, or any
  other production/serve-path file. This tool only *calls* existing
  functions; it does not change them.
- Do not touch `SupabaseHelper` or the `chat` table (out of scope per this
  round's explicit decision).
- Do not add any auto-labeling, heuristic scoring, or "accept without
  `--by`" path. Every golden entry traces to an explicit human
  invocation.
- Do not add a new dependency — `langfuse` is already installed and
  already used by production code.
- Do not build a PII redaction pipeline — no evidence this is a real
  problem yet (Ponytail); human `--skip` covers it if it ever is one.
- Do not wire this script into `Makefile`'s `test`/`lint`/`eval-gates`
  targets or CI — it's a manually-invoked labeling tool, not a gate.

## Required tests (new file: `tests/test_label_eval_candidates.py`)

Mock `get_lf_client`, `retrieve_postgres`, `_structured_claims`,
`validate_and_render` at the module boundary (follow the existing mock
style in `tests/test_retrieval.py` / `tests/test_orchestrator.py` — real
predicate/data shapes, not canned booleans, per this whole program's
established standard). Cover:

1. `--fetch` dedupes on `trace_id` — running it twice against the same
   mocked trace list does not duplicate queue entries.
2. `--fetch` correctly marks `content_available: false` for a 16-hex-char
   `input` and `true` for real text.
3. `--fetch` `as_of` fallback: uses `metadata["as_of"]` when present,
   falls back to `trace.timestamp.date()` with `as_of_source` tagged
   correctly when absent.
4. `--show` and `--label` both raise `SystemExit` on a
   `content_available: false` candidate, without calling the retrieval/
   claims pipeline at all.
5. `--label` rejects an out-of-range `--claim IDX` against a freshly
   mocked (possibly different-length) claims list.
6. `--label` with only `--uris` (no `--claim`) writes to
   `labeled_traffic.json` only; with only `--claim` (no `--uris`) writes
   to `claim_support.json` only; with neither, raises `SystemExit`.
7. `--label` is append-only: seed `claim_support.json` with an existing
   entry, run `--label`, assert the seeded entry is still present
   alongside the new one.
8. `--label` without `--by` raises `SystemExit` (argparse-level or
   explicit check — match `review_documents.py`'s pattern).
9. `--skip` writes no golden entries and does not appear in `--list`
   with default (`pending`) status afterward.
10. `gate_verdict_abstained` in a written `claim_support.json` entry
    matches what the mocked `validate_and_render` actually returned for
    that claim index — not hardcoded.

## Required checks before reporting done

`make test`, `make lint`. `make eval-gates` is not relevant to this task
(no gate/serve-path code changes) — do not report on it either way beyond
noting it's unaffected.

## Commit authorship

Every commit on this branch must be authored as
`Prakash Basnet <basnetprakash090@gmail.com>` — no AI/Claude attribution,
no `Co-Authored-By` trailer.

---

## Rework note (2026-09-04, round 1)

Reviewed `af7d660`. Most of it is correct and matches this brief closely:
the three-step pipeline call, dedup-on-`trace_id`, the hash-content trap
(with two good disclosed additions beyond spec — non-string `trace.input`
refused, `get_lf_client() is None` refused with a clear message — both
kept), `as_of` fallback + tagging, fresh (never-cached) re-runs on every
`--label`, append-only writes, out-of-range claim rejection, and all 10
required tests present and genuinely behavioral (not canned). `make test`
(257 passed, 4 skipped) and `make lint` reproduced myself and match. Also
manually ran `ruff check`/`ruff format --check`/`mypy --strict` on
`scripts/label_eval_candidates.py` directly, since it isn't in the
Makefile's fixed `lint` file list (same pre-existing gap noted for every
other new file across this whole session's history) — clean.

**One disclosed vocabulary change, accepted, not a finding**: `IDX:refutes`
(as written in this brief) was renamed to `IDX:unsupported`, with the old
spelling made a hard error rather than silently accepted — `unsupported`
maps more directly to the stored `supports: bool` field than `refutes`
did. Keep this; do not revert it.

**One real bug, reproduced directly, required fix**: `label()` can reach
`_set_status(trace_id, "labeled")` having written **zero** golden entries
to either file — e.g. `--label t1 --by X --claim "0:skip"` with no
`--uris`: `expected_uris` is empty (nothing written to
`labeled_traffic.json`), the only claim verdict is `"skip"` so the loop's
`continue` means `wrote_claim` never becomes `True` (nothing written to
`claim_support.json`), yet the function falls through to
`_set_status(..., "labeled")` and prints `"labeled"` unconditionally.
Reproduced this exact call directly against the current code this
session: prints `labeled`, neither golden file is created, and the queue
row's `status` becomes `"labeled"` — permanently, since there is no
"unlabel" command. The candidate silently vanishes from
`--list --status pending` with nothing recorded and no way back short of
hand-editing `_traffic_queue.json`. This is a real data-loss footgun in a
tool whose entire purpose is reliably capturing labels — not present in
the original brief's own spec (a gap in this brief's edge-case coverage,
not something to blame on the implementation), but wrong regardless of
who's at fault.

**Required fix**: after computing `expected_uris` and `parsed_claims` in
`label()`, before running the pipeline or writing anything, determine
whether this invocation will actually record anything — `expected_uris`
non-empty, OR at least one parsed claim verdict is `"supports"` or
`"unsupported"` (i.e. not all `"skip"`). If nothing will be recorded,
raise `SystemExit` with a message pointing at `--skip` instead (e.g.
`"nothing to record — every claim was 'skip' and no --uris given; use --skip if this candidate isn't usable"`).
This check can run before the pipeline call (cheap — it's just inspecting
the parsed args), so a no-op `--label` also doesn't need to touch the DB
at all.

**Required test**: add a test proving `--label <id> --by X --claim
"0:skip"` (no `--uris`) raises `SystemExit`, writes neither golden file,
and leaves the queue row's `status` as `"pending"` (not `"labeled"`) —
i.e. the candidate is still recoverable via `--list` afterward.

Same branch, same engineer, per the Rework Loop — not re-scoped. Nothing
else in this diff needs to change.
