# task.md — AGENT-30: exact दफा/धारा/उपदफा/अनुसूची/Act-title lookup

## Objective

`retrieve_postgres()` (`postgres_retriever.py:133`) only ever does vector +
lexical (`plainto_tsquery` bag-of-words) search, fused via `_rrf()`. There
is no structured path for the extremely common case where a user names an
exact legal reference directly — "दफा ९४", "मुलुकी देवानी संहिता को दफा ४५६",
"धारा ५१" — confirmed live gap (finding #6 in the retrieval-quality
review, see `.agent/PROGRESS.md`'s "Retrieval-quality review + 6-task
program" entry). Bag-of-words lexical search over `chunk_text` will often
still surface the right chunk incidentally, but there's no *exact,
deterministic* boost for it, and no path at all for "which Act is this"
questions that don't share vocabulary with the chunk text.

Fix: parse the query for a दफा/धारा-style section number, an अनुसूची
number, and/or a known Act title; if any of those are found, run one more
targeted SQL lookup against `chunks`/`work` (eligibility-gated, same as
every other retrieval branch) and fold its ranked result into the **same**
`_rrf()` fusion vector/lexical already use — not a bypass lane, not a
separate gate, just one more ranked list into the existing fusion.

## Schema grounding — read before writing code

- `chunks.section_number` holds the दफा/धारा number for `level='section'`
  chunks. `chunks.parent_section` holds the *same* दफा number for
  `level='subsection'|'proviso'` chunks under it (confirmed via
  `laws_chunker.py`'s `LawChunk.parent_section` docstring: "दफा number when
  level='subsection'/'proviso'"). **There is no separate उपदफा-number
  column** — subsections are chunked for size reasons, not individually
  numbered in a queryable field. This means `उपदफा` in a query is a signal
  to look for (parse it), but it does not add filtering precision beyond
  the दफा number it's nested under — do not invent a subsection-number
  match that the schema can't back. Say this plainly in your completion
  report, don't silently pretend otherwise.
- `धारा` (used specifically for Constitution-type works) is stored the same
  way as `दफा` — a numeral in `section_number`/`parent_section`. The
  distinction between the two words is about which document uses which
  term, not a separate schema field. Treat both keywords as introducing the
  same kind of number for matching purposes.
- `अनुसूची` (schedule) has **no dedicated column for ordinary Acts** — only
  tariff-schedule Acts (`भन्सार_महसुल_ऐन_२०८१`-style) model schedules
  structurally, via the separate `TariffChunk`/`tariff_chunker.py` system
  (`level='tariff_heading'|'tariff_row'|'tariff_note'`, its own
  `part_number`/`heading_code`/`subheading_code` fields) — **do not touch
  that system, out of scope**. For an ordinary Act's अनुसूची mention, the
  only thing available is a literal-text match: does `chunk_text` contain
  the string "अनुसूची <N>"? Build the अनुसूची signal as exactly that — a
  `strpos()` containment check on `chunk_text`, not a structured-column
  match. This is genuinely lower-precision than the दफा/धारा path; say so.
- `chunks.work_id` (UUID, `REFERENCES work(id)`) already exists directly on
  the `chunks` row — Act-title filtering does not need to go through
  `component`/`component_uri` at all, just `chunks.work_id`.
- `work.title_ne` is the canonical Act title (per AGENT-28's established
  direction — resolve through canonical tables, not per-chunk denormalized
  copies).

## Acceptance criteria

1. **Query parsing** — new, separately testable functions (not buried
   inline in `retrieve_postgres`):
   - `_parse_section_reference(query: str) -> str | None` — matches
     `(?:दफा|धारा)\s*(\d+)` and returns the number as a string, or `None`.
     Also match `उपदफा\s*\(?(\d+)\)?` if present (parse it — needed so the
     completion report can note the schema limitation above with a real
     example — but per the schema grounding, it does not add its own
     filter condition; the section number it's nested under is what
     actually filters).
   - `_parse_schedule_reference(query: str) -> str | None` — matches
     `अनुसूची\s*(\d+)` and returns the number, or `None`.
   - Both run on the query **after** the existing `_preprocess()` call
     (`retrieve_postgres` already NFC-normalizes and digit-folds to ASCII
     before any retrieval logic runs — reuse that, do not re-digit-fold or
     handle Devanagari numerals again in these new functions; match on
     `\d+` directly).
2. **Act-title resolution** — new function
   `_resolve_act_title(conn: connection, query: str) -> str | None`,
   returns the matched `work.id` as text or `None`:
   ```sql
   SELECT id::text, title_ne FROM work WHERE strpos(%(query)s, title_ne) > 0
   ORDER BY length(title_ne) DESC LIMIT 1
   ```
   Use `strpos()` (plain substring search), **not** `ILIKE` with
   string-concatenated wildcards — `title_ne` is untrusted-shaped input
   flowing into a pattern position; `strpos` sidesteps LIKE-wildcard
   injection entirely rather than needing `ESCAPE` handling for a
   near-zero-probability case. Longest-title-first ordering picks the more
   specific match when one Act's title happens to be a substring of
   another's. Runs unconditionally (cheap, single query against the small
   `work` table) — an Act-title-only query with no section number (e.g.
   "भन्सार महसुल ऐन २०८१ के हो?") is a legitimate case per this task's own
   scope, not an edge case to skip.
3. **Exact lookup query** — new closure `exact_lookup_search()` alongside
   the existing `vector_search`/`lexical_search` closures (same
   `with conn.cursor() as cur:` block, same connection, no new transaction
   scope), **only invoked when at least one of `section_num`,
   `schedule_num`, `act_work_id` is not `None`** — skip the query entirely
   otherwise, don't waste a round-trip on the common case where none of
   this applies:
   ```sql
   SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
          c.chunk_type, c.section_number, d.source_id
   FROM chunks c
   JOIN documents d ON d.id = c.document_id
   WHERE c.id::text = ANY(%(eligible)s)
     [AND c.work_id = %(work_id)s]                              -- if act_work_id
     [AND (c.section_number = %(num)s OR c.parent_section = %(num)s)]  -- if section_num
     [AND strpos(c.chunk_text, 'अनुसूची ' || %(schedule_num)s) > 0]     -- if schedule_num
   LIMIT %(limit)s
   ```
   Same 8-column shape as `vector_search`/`lexical_search` (reuses the
   existing `_hit()` unpacking and the existing `documents` join — no
   changes needed to `_hit()`). Same `eligible` filter as every other
   branch — Core Invariant #2, no path skips the gate. Same `limit` (`k *
   3`) already computed for the other two searches — don't invent a new
   constant.
4. **Merge into the existing fusion, not a bypass**: add the exact-lookup
   chunk-id list as one more entry in `ranked_lists` before the existing
   `_rrf(ranked_lists)` call (`postgres_retriever.py:251`). Add its rows to
   the `rows` dict (`postgres_retriever.py:238-241`) the same way
   `vector_rows`/`lexical_rows` already are — this is required, not
   optional: a chunk id that appears **only** in the exact-lookup list
   would otherwise cause a `KeyError` at `rows[chunk_id]` when building
   `candidates` (`postgres_retriever.py:262`). No new relevance threshold,
   no new gate, no bypass of `rerank()` — exact matches flow through
   exactly the same `_RELEVANCE_THRESHOLD` → `rerank()` pipeline as
   everything else (an exact match ranked #1 in its own list already scores
   `1/(60+1) ≈ 0.0164` under RRF, well above the `0.005` threshold, so no
   special-casing is needed for it to survive the gate on its own).
5. **Observability**: add one more `_end_span(retrieval_span,
   "exact_lookup", ...)` call, mirroring the existing
   `vector_search`/`lexical_search` spans (candidate count, whether it ran,
   latency) — matches the established pattern in this exact function, cheap
   to add, keep the trace complete.
6. No change to `vector_search`, `lexical_search`, `_rrf()`, `_hit()`,
   `rerank()`, the relevance-threshold gate, or the final re-fetch/render
   block (`postgres_retriever.py:292-323`) — this task only adds one more
   input to the existing fusion.

## Branch

`agent/exact-citation-lookup` (already created off `dev`, in sync).

## Governing design references

- `AGENTS.md` — non-negotiables: eligibility gate on every path, same
  predicate every branch.
- `system-design.md` §8 Query plane: "...parallel retrieval over eligible
  set (exact URI · title · citation tokens · BM25 Nepali · multilingual
  vector · variants) → fusion (RRF) → rerank..." — this task is exactly the
  "exact URI · title · citation tokens" arm the design already names but
  that isn't built yet.
- `system-design.md` Core Invariant #2 (deterministic eligibility gate
  runs pre-retrieval, on every branch, same predicate every path) — the
  new exact-lookup query must filter by the same `eligible` set as
  `vector_search`/`lexical_search`.

## Allowed scope

- `app/retrieval/postgres_retriever.py` — the new functions/closure
  described above, and the minimal wiring into `retrieve_postgres()`
  (parsing calls, `ranked_lists`/`rows` merge, new span). Do not touch
  `_hit()`, `_rrf()`, `translate_query`, `_embed_query`, or the reranking/
  final-refetch block.
- `tests/test_retrieval.py` — the existing test file for this module.
  Extend its `Cursor`/`Conn` mock (currently dispatches on `"embedding
  <=>"` vs. `"ts_rank_cd"` substrings in the executed SQL — add a third
  branch for the exact-lookup query's SQL shape) rather than building a
  parallel mock; check `patch_common()` and the existing tests before
  writing new ones so you don't regress the vector/lexical dispatch.
- `.agent/PROGRESS.md` — do not edit; Claude updates this after review.

## Explicitly forbidden

- Do not touch `tariff_chunker.py`, `TariffChunk`, or any tariff-schedule
  ingestion code — schedule lookup for ordinary Acts is a lexical
  `chunk_text` containment check only, per the schema grounding above; a
  structured schedule model is a separate, larger, ingestion-side task.
- Do not add a subsection-number column, migration, or any schema change —
  Ponytail-gated, out of scope. If the lack of a उपदफा-number column
  genuinely blocks something essential, stop and report it rather than
  adding a migration yourself.
- Do not bypass `_RELEVANCE_THRESHOLD` or `rerank()` for exact matches —
  they flow through the same pipeline as everything else, per acceptance
  criterion #4.
- Do not add a new dependency, service, or embedding call for this — pure
  SQL + regex, deterministic, no LLM.
- Do not touch `validation_gate.py`, `gated_orchestrator.py`,
  `query_graph.py`, or anything precedent-related.

## Required checks

- `make test`
- `make lint`
- `make eval-gates` — all three zero-tolerance gates must stay at `0`,
  untouched by this task, must show no regression.

## Zero-tolerance gates guarded

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

## Self-review before returning

- Confirm the exact-lookup query is filtered by `eligible` exactly like
  `vector_search`/`lexical_search` — no path that reads `chunks` without
  the eligibility filter.
- Confirm a chunk id present **only** in the exact-lookup results (not in
  vector or lexical results) survives all the way through
  `retrieve_postgres()` without a `KeyError` — write a test that seeds
  exactly this case (a exact-only hit, empty vector/lexical arms) and
  asserts it appears in the final output.
- Confirm `_parse_section_reference`/`_parse_schedule_reference` are pure
  functions with no DB access, and that they correctly return `None` (not
  crash, not empty string) for ordinary queries with no legal reference in
  them at all.
- Confirm `_resolve_act_title`'s `strpos()` query is exercised by a test
  proving the longest-match-wins behavior when one Act's title is a
  substring of another's (seed two `work` rows where this is true).
- Report the queries you had to add to the `Cursor`/`Conn` test mock's
  dispatch logic and why the existing vector/lexical dispatch still works
  unchanged.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — no AI attribution, no `Co-Authored-By:
Claude` trailer, no "Generated with Claude" line, in any commit on this
branch. Use `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
