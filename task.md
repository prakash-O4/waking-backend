# task.md — AGENT-45: add input/output visibility to the remaining blind Langfuse spans

## How this was found

While debugging AGENT-42/43 with real traces, Prakash flagged that
`co_retrieve_parent_resolution`, `cross_ref_resolution`, and
`enabling_power_resolution` show up in Langfuse with no `input`/
`output` at all — just a bare count in `metadata` — and asked "what do
they even do, are they there just for vibes?" Investigation (grounded
in the actual code, not guessed) confirmed all three are real,
narrowly-scoped mechanisms — not vestigial — but the trace genuinely
gives zero visibility into what they looked at or found, which is
exactly the same gap the `retrieval` span had until it was fixed today
(commit `e4457cf`, "Add input/output to the Langfuse retrieval span").
Prakash then pointed out this same blind-span pattern also covers the
retrieval sub-stages (`stage.eligibility_gate` through `stage.rerank`)
and asked for it fixed comprehensively. Widening the sweep found two
more: `authority_ranking` and `validation` have the identical gap.

**Full list of spans in scope** (12 total — everything that currently
creates a span with `metadata` only, no `input`/`output`, excluding the
already-fixed `retrieval` span itself):

In `app/retrieval/postgres_retriever.py`, all created via the shared
`_end_span()` helper (`postgres_retriever.py:64-74`):
1. `stage.eligibility_gate`
2. `stage.vector_search`
3. `stage.lexical_search`
4. `stage.exact_lookup`
5. `stage.relevance_gate`
6. `stage.rrf_fusion`
7. `stage.rerank`

In `app/retrieval/query_graph.py`, each created inline with its own
`start_observation()`/`.update()`/`.end()`:
8. `authority_ranking` (line 228)
9. `co_retrieve_parent_resolution` (line 253)
10. `cross_ref_resolution` (line 282)
11. `enabling_power_resolution` (line 396)
12. `validation` (line 426)

## Objective

Add `input`/`output` to each span above so a real trace shows what
each stage actually considered and decided — not just a count — while
staying consistent with PS-14 ("traces store IDs/hashes by default;
raw content in a separate short-retention store") and the precedent
already set by the `retrieval` span fix: short, capped, ID/score-level
detail, never full raw chunk text, never the full `eligible` id list
(which runs into the thousands).

## Part A — `app/retrieval/postgres_retriever.py`: extend `_end_span()`

Current helper (`postgres_retriever.py:64-74`):
```python
def _end_span(trace: Any, stage: str, **metadata: Any) -> None:
    """Create a span and immediately end it so Langfuse records endTime."""
    if trace is None:
        return
    try:
        span = trace.start_observation(
            name=f"stage.{stage}", as_type="span", metadata=metadata
        )
        span.end()
    except Exception:
        pass
```

Change to accept optional `input`/`output`, passed through only when
given (don't pass `input=None`/`output=None` explicitly — omit the
kwarg entirely when not provided, matching how every other span in
this file already behaves when it has no input):
```python
def _end_span(
    trace: Any,
    stage: str,
    *,
    input: Any = None,
    output: Any = None,
    **metadata: Any,
) -> None:
    """Create a span and immediately end it so Langfuse records endTime."""
    if trace is None:
        return
    try:
        kwargs: dict[str, Any] = {
            "name": f"stage.{stage}",
            "as_type": "span",
            "metadata": metadata,
        }
        if input is not None:
            kwargs["input"] = input
        span = trace.start_observation(**kwargs)
        if output is not None:
            span.update(output=output)
        span.end()
    except Exception:
        pass
```

Then, at each of the 7 call sites (all inside `retrieve_postgres()`),
add `output=` (and `input=` where noted) using **only** `component_uri`
(chunk id), `section_number`, and score — never `chunk_text` — capped
to a small sample (3-5 items) so traces stay light:

1. **`eligibility_gate`** (`postgres_retriever.py:287-293`) — leave
   as-is, metadata-only. The `eligible` list is thousands of bare UUIDs
   with no other structure — dumping it or a sample of it adds no real
   debugging value over the existing `eligible_count`. Do not add
   input/output here; note in the PR/commit why, so this isn't mistaken
   for an oversight later.

2. **`vector_search`** (`postgres_retriever.py:401-409`) — add
   `output` = the top 3 rows from `vector_rows` (and `vector_rows_ne`
   if present), each as `{"chunk_id": row[0], "section_number": row[6],
   "score": round(float(row[-1]), 4)}`.

3. **`lexical_search`** (`postgres_retriever.py:419-425`) — same shape,
   top 3 from `lexical_rows`/`lexical_rows_ne` when `lexical_ran`.

4. **`exact_lookup`** (`postgres_retriever.py:432-438`) — add `input` =
   `{"section_range": section_range, "section_num": section_num,
   "section_nums": section_nums, "schedule_nums": schedule_nums,
   "act_work_ids": act_work_ids, "proviso_ref": proviso_ref}` (this is
   the actual reason exact-lookup did or didn't fire — currently
   invisible) and `output` = top 3 rows from `exact_rows` as `{"chunk_id":
   row[0], "section_number": row[6]}`.

5. **`rrf_fusion`** (`postgres_retriever.py:462-468`) — add `output` =
   top 5 from `rrf_scores` as `{"chunk_id": chunk_id, "rrf_score":
   round(score, 4)}`.

6. **`relevance_gate`** (`postgres_retriever.py:476-482`) — add
   `output` = top 3 from `candidates` as `{"chunk_id":
   c["component_uri"], "score": round(c.get("score", 0.0), 4)}`.

7. **`rerank`** (`postgres_retriever.py:495-502`) — add `output` = top
   3 from `ranked` as `{"chunk_id": r["component_uri"], "score":
   round(r.get("score", 0.0), 4)}`.

## Part B — `app/retrieval/query_graph.py`: 5 inline spans

Add `input`/`output` directly to each existing
`start_observation()`/`.update()` call — same capped, ID/score-only
style, no new helper needed (5 call sites, each already self-contained
— extracting a shared helper here would be over-engineering for this
size; keep the smallest local change).

8. **`authority_ranking`** (`query_graph.py:221-233`) — add `output` =
   top 3 of `ranked` as `{"chunk_id": h.get("component_uri"), "tier":
   h.get("tier")}`.

9. **`co_retrieve_parent_resolution`** (`query_graph.py:236-262`) — add
   `output` = `additional` mapped to `{"chunk_id":
   h.get("component_uri"), "section_number": h.get("section_number")}`
   (full list is fine here — `additional` is rarely more than a couple
   items given how narrowly this fires; no cap needed, but do not
   include `text_ne`/raw content).

10. **`cross_ref_resolution`** (`query_graph.py:265-291`) — same shape
    as #9, mapped over `additional`.

11. **`enabling_power_resolution`** (`query_graph.py:372-405`) — same
    shape as #9, mapped over `additional`.

12. **`validation`** (`query_graph.py:408-440`) — add `output` =
    `all_results` mapped to `{"evidence_id": r.get("evidence_id"),
    "abstained": r.get("abstained"), "as_of": r.get("as_of")}` — no
    quote/claim text, just the pass/fail per claim so a trace shows
    *which* claims were rejected, not just the count.

## Acceptance criteria

- All 12 spans listed above show meaningful `input`/`output` in a real
  trace (except `eligibility_gate`, deliberately left as-is per Part A
  item 1).
- No span output includes `chunk_text`, `text_ne`, or any other raw
  legal-text field — IDs, section numbers, and scores only. No span
  output includes the full `eligible` id list.
- `stage.*` spans keep their existing `metadata` fields unchanged —
  this task only adds `input`/`output` alongside them, never removes or
  renames an existing metadata key (anything currently reading
  `top_score`, `candidate_count`, etc. from these spans must keep
  working).
- `_end_span()`'s new `input`/`output` parameters are keyword-only
  (`*,` in the signature) so no existing positional-metadata call site
  can accidentally break.
- No change to retrieval logic, gate logic, or what gets returned from
  `retrieve_postgres()`/any node function — this task only adds
  observability, it does not change behavior.
- `make test`/`make lint` — use `.venv/bin/python`, not bare `python3`.
- Manual check: run one real query against the live server/console,
  pull the resulting trace, confirm all 11 non-eligibility spans now
  show populated `input`/`output`.

## Explicitly forbidden

- No new dependency, no new tracing abstraction/helper class beyond
  the one extended function (`_end_span`) — this is additive
  instrumentation only.
- Do not touch `eligibility_gate`'s span (see Part A item 1 for why).
- Do not touch the already-fixed `retrieval` span
  (`postgres_retriever.py:264-283, 531-547`) or its existing
  input/output shape.
- Do not touch any gate/temporal/precedent logic, any SQL query, any
  ranking/reranking algorithm, or the shape of any function's return
  value — every change in this task is purely additive tracing
  metadata.
- Do not add raw `chunk_text`/`text_ne` to any new span output — PS-14
  requires traces store IDs/hashes by default, not raw content.

## Governing references

PS-14 ("Traces store IDs/hashes by default; raw content in a separate
short-retention, dual-control store; redaction at collector") is the
one PS requirement directly relevant here — every addition in this
task must stay ID/score-level, never raw text. No Core Invariant is
touched; this is pure observability.

## Required checks

- `.venv/bin/python -m pytest tests/`
- `.venv/bin/python -m ruff check` / `ruff format --check` / `mypy
  --strict` on the exact Makefile file list (substitute
  `.venv/bin/python` for `python3`)
- Manual live trace check as described above

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, Pi, or any AI
attribution. Enforce via:
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`
No `Co-Authored-By` trailers, no "Generated with" lines.
