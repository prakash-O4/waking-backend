# task.md — AGENT-32: confirmed production gaps (as_of propagation, interruption, parallel fan-out, live-corpus gate, degraded-mode labeling, README correction)

## Objective

Six confirmed, code-verified gaps from a production-readiness review, bundled
into one task per Prakash's explicit instruction (all in one go, not phased).
Each was independently verified against actual code/ADR text before being
accepted — see `.agent/PROGRESS.md`'s dated entry for the full grounding.
Two design decisions (streaming scope, connection-pool approach) were
resolved with Prakash directly before this brief was written — do not
re-litigate them.

This task runs after AGENT-31 (stress suite) merges — different files,
no overlap, but sequenced so the program's tracker stays linear.

## 1. Per-claim as_of leak in co-retrieval resolvers

**Gap**: `co_retrieve_parent_resolver_node`, `cross_ref_resolver_node`
(`query_graph.py:128-163`) call their orchestrator functions with
`state["session_as_of"]` — a single scalar — even though `state["all_hits"]`
can contain hits from multiple issues at different declared as-of values
(diachronic/comparative queries). `enabling_power_resolver_node` (`:244`)
has the same bug in its own per-hit loop. Core Invariant #6: "every claim
validates against its declared as-of... comparative queries carry a set of
as-of points, one per claim."

**Fix — do not change `_resolve_co_retrieve_parents`, `_resolve_cross_refs`,
or `_fetch_enabling_chunk`'s signatures or existing unit tests.** Fix this
entirely at the node level in `query_graph.py`:
- `co_retrieve_parent_resolver_node` / `cross_ref_resolver_node`: group
  `state["all_hits"]` by `_issue_idx`, resolve each group's true as_of via
  `state["issue_queries"][idx]["as_of"]` (fall back to `session_as_of` for
  the single-issue-query default case, matching the existing `issue_queries
  or [{"query": ..., "as_of": state["session_as_of"], ...}]` pattern used
  elsewhere in this file). Call the existing orchestrator function once per
  distinct as_of value present (not once per hit — memoize/dedupe when
  multiple issues share the same as_of, the common non-diachronic case, so
  this doesn't add cost for ordinary single-as_of queries). Merge results
  back, preserving `_issue_idx` on each added hit exactly as today.
- `enabling_power_resolver_node`: resolve `as_of` per-hit inside its
  existing `for h in hits[:5]` loop instead of using one `state["session_as_of"]`
  for all of them.

## 2. Fact-extractor interrupts too eagerly

**Gap**: `fact_extractor_node` sets `interrupted = bool(required)`
(`query_graph.py:29-30`) with no check of whether retrieval could still
answer generally. The actual ADR (`docs/adr-001-multi-agent-query-architecture.md:216`)
specifies: `any required=True AND no retrieved results cover it → interrupt`
— the code drops the second clause entirely.

**Fix**: when `required` facts exist, before interrupting, run a bounded
coverage probe: `_orch.retrieve_postgres(conn, state["raw_query"],
state["session_as_of"], k=3, lf_trace=lf_trace)` — only if
`not _orch._wall_clock_expired(state["wall_clock_start"])` (if the wall
clock is already tight, skip the probe and keep today's safe default:
interrupt). `fact_extractor_node` needs `conn` from `config["configurable"]["conn"]`
(not currently read there — small, additive extraction, same pattern every
other node already uses). If the probe returns at least one hit, do not
interrupt — **and** relabel that fact's `type` from `"required"` to
`"clarifying"` in the returned `missing_facts` list before returning state.
This matters: `_compose_answer`'s prompt (`gated_orchestrator.py:198-204`)
filters `missing_facts` to `type in ("clarifying", "informational")` only
— a `required`-but-covered fact would otherwise silently vanish from the
final answer's documented caveats instead of being surfaced like any other
clarifying fact. If the probe returns nothing, interrupt exactly as today.

## 3. Retrieval is not actually parallel

**Gap**: ADR (`:78-82`): "Node 2 — Parallel Retriever (fan-out, one per
issue), Fan-out cap: 5 parallel branches maximum," with connection-pool
sizing explicitly flagged as a pre-deploy review item (`:356,366`).
`retrieve_node` (`query_graph.py:47-69`) is a plain sequential `for` loop.

**Design already decided with Prakash**: add a small, bounded connection
pool sized to the ADR's own fan-out cap (5) — not an unreviewed new
mechanism, the ADR already named and sized it.

**Fix**:
- New small pool helper (e.g. `app/retrieval/db_pool.py`): a lazily-created
  `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn)`. Reuse
  the exact DSN-resolution logic `app/authority/writer.py::connect()`
  already uses (`DATABASE_URL` or `SUPABASE_DB_URL`) — extract it into a
  tiny shared helper both `connect()` and the new pool call, rather than
  duplicating the two-line lookup (Ponytail: reuse, don't duplicate).
- `retrieve_node`: when `len(issue_queries) > 1`, use
  `concurrent.futures.ThreadPoolExecutor(max_workers=min(len(issue_queries), 5))`
  to run one `retrieve_postgres(pool_conn, iq["query"], iq["as_of"], ...)`
  per issue concurrently — each thread borrows its own connection via
  `pool.getconn()` / returns it via `pool.putconn()` in a `finally`. Build
  `all_hits` in the same deterministic issue-index order as today (collect
  per-issue results into a list indexed by `idx`, don't append in
  whichever-thread-finishes-first order) — do not silently change hit
  ordering behavior. For a single issue query, or if pool creation fails,
  fall back to today's sequential path using the existing `conn` — no
  regression for the common single-issue case, no hard dependency on the
  pool always being available.
- **The pool is request-scoped, not app-lifetime**: create it fresh inside
  `retrieve_node` sized to `min(len(issue_queries), 5)`, close it
  (`closeall()`) in a `finally` after the fan-out completes. Do not add
  FastAPI startup/shutdown hooks or a global pool — smallest local change,
  not a connection-reuse-across-requests optimization (that's a separate,
  bigger change if ever wanted).
- The graph's main connection (passed into `run_query`/`config["configurable"]["conn"]`)
  is untouched and continues to serve every other node — this fix is
  scoped entirely inside `retrieve_node`.
- Self-review requirement: verify Langfuse span creation
  (`_end_span`/`_lf_gen_start` etc.) is safe under concurrent invocation
  from multiple threads now that `retrieve_postgres` can run in parallel —
  don't assume, check the client library or serialize span calls if there's
  real doubt.

## 4. Eval evidence is thin — live-corpus zero-tolerance check

**Gap**: `app/eval/gates.py`'s three checks (`check_repealed_as_current`,
`check_not_yet_effective_as_current`, `check_overruled_as_good_law`)
connect to the live DB but each **deletes** the real `lifecycle_effect`
rows for one arbitrary component, inserts a synthetic effect, checks
`is_eligible()`/`is_good_law()` against it, then rolls back via savepoint
(`:50-70`). This proves the isolated predicate function behaves correctly
on a manufactured case — it never checks whether the corpus's actual,
currently-approved data has zero real violations.

**Design decision made here, not open for reinterpretation**: this task
adds an **engineering** check (an independent live-corpus regression
query), not new legal golden-data labeling — expanding golden QA sets
responsibly needs Prakash's own legal verification, out of scope for an
engineer to do unsupervised.

**Fix**: add new functions to `app/eval/gates.py` —
`check_repealed_as_current_live()` / `check_not_yet_effective_as_current_live()`
— that run an **independent** SQL query (not via `eligible_chunk_ids()`/
`is_eligible()` themselves, to avoid circularity — a genuinely separate
verification path) directly against `chunks` + `lifecycle_effect`:
count how many chunks currently returned by `eligible_chunk_ids(conn,
date.today())` have an approved `repeal`/`expiry`/`declared_invalid`
effect with `lower(legal_valid_time) <= today`, or lack a covering
approved `commence` effect. Both counts should be `0`. Add both to
`main()`'s printed output and its exit-code check (`repealed-as-current`
alongside a new `repealed-as-current-live-corpus` line, etc.) — one
command, one report, strictly more rigorous than today, not a second
fragmented gate command.

## 5. Silent degraded-mode fallbacks

**Gap**: `reranker.py::rerank()` has three tiers (Cohere → FlashRank →
passthrough) with no signal to the caller of which one actually ran. The
existing Langfuse span in `postgres_retriever.py` (`ran=bool(get_settings().COHERE_API_KEY)`)
reports whether Cohere is *configured*, not whether it *succeeded* — even
the trace is inaccurate. The reasoner's extractive fallback does reach
`_response.query_type` today, but only as one coarse flag for the whole
multi-issue answer.

**Fix — additive only, no signature changes** (confirmed via `grep`: the
only production caller of `reranker.rerank` is `postgres_retriever.py`;
`retrieve_postgres`'s own return type has five callers across `app/eval/*`
and must not change):
- `rerank()`: tag each hit dict it returns with `hit["reranker_tier"] =
  "cohere" | "flashrank" | "passthrough"` before returning — same
  `list[dict[str, Any]]` return type, richer dict contents. Fix the
  Langfuse span in `postgres_retriever.py` to report the actual tier used,
  not config-key presence.
- `postgres_retriever.py`: when building `result_hits` from `ranked` in the
  final re-fetch block (`:292-323`), carry `reranker_tier` forward (a
  simple dict-merge on top of what `_hit()` returns — no `_hit()` signature
  change needed).
- `answer_composer_node` (`query_graph.py`): compute `degraded_mode:
  list[str]` by scanning `state["all_hits"]` for any `reranker_tier` !=
  `"cohere"` and `state["query_type"] == "extractive"`, producing entries
  like `["reranker_fallback:flashrank"]` / `["reasoner_fallback:extractive"]`.
  Attach this list to `_response` on **both** paths — the composed-success
  path and the `composed is None` fallback path — so degraded-mode is
  visible either way, not just when the composer succeeds.

## 6. README claims outrun implementation

**Design decision made here, not open for reinterpretation**: fix the
docs, do not build streaming. `/ask` (`app/main.py:44-59`) returns a plain
dict/`JSONResponse` — confirmed via `grep`, zero `StreamingResponse`/SSE
usage anywhere in `app/`. `token_buffer.py` exists on disk but is not
imported by `main.py` — dead code the README describes as live.

**Fix**: edit `README.md` to describe what `/ask` actually does today (a
single JSON response per request) — remove or clearly mark as
not-yet-implemented the "real-time streaming responses," "SSE Streaming,"
and `token_buffer.py`-as-active-infrastructure claims. Do not delete
`token_buffer.py` itself (out of scope — it may be genuinely intended for
a future streaming feature; this task corrects documentation accuracy, it
does not make a call on that file's fate).

## Branch

`agent/confirmed-production-gaps` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: eligibility gate on every path; every
  claim validates against its declared as-of; abstention is server-owned.
- `docs/adr-001-multi-agent-query-architecture.md` — Node 2 fan-out spec
  (`:78-82`), missing-facts routing (`:213-230`), connection-pool sizing
  flagged as pre-deploy risk (`:356,366`).
- `system-design.md` Core Invariant #6 (per-claim as-of, comparative
  queries carry a set), §14 PS-12 (stress/eval gates backed by numbers).

## Allowed scope

- `app/retrieval/query_graph.py` — `co_retrieve_parent_resolver_node`,
  `cross_ref_resolver_node`, `enabling_power_resolver_node`,
  `fact_extractor_node`, `retrieve_node`, `answer_composer_node`. Do not
  touch `validate_node`, `reasoner_node`, or graph wiring beyond what's
  needed for the above.
- `app/retrieval/postgres_retriever.py` — reranker-tier threading and span
  accuracy only. Do not touch retrieval logic, RRF, or the eligibility
  gate.
- `app/retrieval/reranker.py` — tag hits with `reranker_tier` only.
- New `app/retrieval/db_pool.py` (or equivalent small module).
- `app/authority/writer.py` — only to extract the shared DSN-resolution
  helper `connect()` already contains, for reuse by the new pool.
- `app/eval/gates.py` — new live-corpus check functions + `main()` output.
- `README.md` — streaming/SSE claims only.
- `tests/` — updates/additions in `tests/test_orchestrator.py`,
  `tests/test_retrieval.py`, `tests/test_eval_gates.py` (new file if none
  exists — check first), and any new test files needed for the pool/
  fan-out logic.
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not touch `app/retrieval/advanced_retriever.py` or
  `app/retrieval/retrieval_orchestrator.py` — confirmed dead/legacy code
  (Pinecone-based, not wired into `/ask` at all, `main.py` imports
  `gated_orchestrator.answer` exclusively). Not in scope, don't "clean it
  up" as a drive-by.
- Do not build real SSE/streaming — decided, docs-only fix for item 6.
- Do not add a global/app-lifetime connection pool or FastAPI
  startup/shutdown hooks — request-scoped pool only, per item 3's design.
- Do not change `_resolve_co_retrieve_parents`, `_resolve_cross_refs`,
  `_fetch_enabling_chunk`, or `retrieve_postgres`'s function signatures —
  fixes are additive/caller-side only, per items 1 and 5.
- Do not expand golden eval datasets (`app/eval/golden/*.json`) — item 4 is
  an engineering check, not new legal data labeling.
- Do not touch `tests/stress/` (AGENT-31's territory, already merged by
  the time this starts) beyond what naturally still passes.

## Required checks

- `make test`
- `make stress` (must still pass — this task must not regress AGENT-31's
  suite, especially the romanized-eligibility and enabling-power cells,
  which touch the same nodes items 1/3/5 modify)
- `make lint`
- `make eval-gates` (all three zero-tolerance gates at `0`, plus the two
  new live-corpus checks from item 4, also at `0`)

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`
- (new) live-corpus variants from item 4, also `= 0`

## Self-review before returning

- Item 1: prove with a seeded test that two hits from different issues
  (different as_of, one where a candidate co-retrieve-parent/cross-ref is
  eligible and one where it isn't at that issue's *own* as_of) are
  resolved correctly and independently — not both evaluated against one
  shared as_of.
- Item 2: prove the coverage-probe path with a test where the probe
  returns hits (no interrupt, fact relabeled to `clarifying`) and one
  where it returns nothing (interrupt, unchanged from today).
- Item 3: prove `all_hits` ordering is unaffected by concurrent execution
  (deterministic, matches sequential-mode output for the same inputs).
  Confirm the pool is properly closed even when a thread raises.
- Item 4: confirm the new live-corpus checks are genuinely independent of
  `eligible_chunk_ids()`'s own SQL (a bug in that function's query
  wouldn't make this check pass by construction).
- Item 5: confirm `reranker_tier` survives from `rerank()` through to
  `_response.degraded_mode` in an end-to-end test, not just at the
  `reranker.py` unit level.
- Item 6: confirm the README no longer claims streaming/SSE behavior that
  `/ask` doesn't have.
- Report explicitly which of the six items, if any, turned out smaller or
  larger than scoped here once you were in the code — don't silently
  under- or over-deliver against this brief.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
