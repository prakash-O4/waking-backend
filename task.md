# task.md — AGENT-35: legal reference parser improvements

## Objective

Post-AGENT-34 roadmap item 3. Extend AGENT-30's exact दफा/धारा/उपदफा/
अनुसूची/Act-title lookup in `app/retrieval/postgres_retriever.py` to
handle: (1) section ranges, (2) Act aliases/colloquial names + a real
year-suffix matching bug, (3) multiple Acts referenced in one query, (4)
provisos referenced by name. Bounded, no open design questions beyond the
one already resolved with Prakash (alias data source — see below).

This is purely a retrieval-precision improvement inside the existing exact-
match lane, which already feeds the same RRF fusion as vector/lexical, no
bypass, no skipped relevance threshold, no skipped rerank (AGENT-30's own
constraint, unchanged, still applies). No ingestion, gate, temporal, or
authority-store path is touched — PS-check confirms no PS-1..18 item is
materially affected; this stays entirely inside query parsing and one
retrieval SQL builder.

## Grounding (confirmed this session by reading code — no live DB this
session, see "Must verify against live DB before relying on it" below;
don't re-derive)

- Current state (`app/retrieval/postgres_retriever.py`):
  - `_parse_section_reference(query)` — `re.search(r"(?:दफा|धारा)\s*(\d+)", query)`,
    single number, first match wins (leftmost).
  - `_parse_subsection_reference(query)` — parses उपदफा but is **never
    called** from `retrieve_postgres()` (confirmed dead code, matches
    AGENT-30 round 2's own review note — leave it exactly as is, don't
    "fix" it into being called unless it's actually needed by this task,
    which it isn't).
  - `_resolve_act_title(conn, query)` — `strpos(%(query)s, title_ne) > 0`,
    i.e. **the query must contain the Act's full canonical `title_ne`
    verbatim, including the trailing year clause**, to match at all.
    `LIMIT 1` — single Act only.
  - `exact_lookup_search()` builds an ANDed `filters: list[str]` — section
    number equality, schedule-number substring, `work_id` equality — and
    is one lane feeding `_rrf()` alongside vector/lexical. This
    accumulator pattern is what every change below plugs into; no
    structural rewrite needed.
- `chunks.section_number` and `chunks.parent_section` are `TEXT`
  (`migrations/005_ingestion_pipeline.sql:90,94`), populated by
  `app/ingestion/laws_chunker.py::_DAFA_HEADING_RE = re.compile(r"\*\*([०-९]+)\.\s*([^\n*]+?)\*\*")`
  — **Devanagari digits only, no letter-suffix support** (e.g. no "५क"
  parsing exists anywhere in the chunker). So today's stored values are
  plain digit strings — but confirm this holds for the live corpus before
  relying on it (see verification list below); this is a chunker-level
  limitation, not something this task fixes.
- `chunks.level` (not `chunk_type`) is the real structural marker for
  provisos. Traced end-to-end: `laws_chunker.py::_emit_subsection` (line
  182) detects a `स्पष्टीकरण` block via `_PROVISO_RE`, splits it into an
  operative-clause chunk (`level="subsection"`) and a proviso chunk
  (`level="proviso"`, PS-16 co-retrieval already wired via
  `co_retrieve_parent_index`) — both share the same `section_number`
  (their parent दफा's number, per `_emit`'s call convention: subsection/
  proviso chunks pass the दफा number as **both** `section_number` and
  `parent_section`). `app/ingestion/pgvector_indexer.py::_chunk_row()`
  persists `"level": chunk.level` directly to `chunks.level`
  (`_CHUNK_INSERT_SQL` includes `level` as its own column — this is
  **not** the same as the human-readable `chunk_type` field, which for
  law chunks is derived separately as `f"दफा {section_number}"` and does
  **not** distinguish proviso from operative text).
- Golden data (`app/eval/golden/phase_c_romanized.json`) confirms
  `work.title_ne` values carry **Devanagari-digit** year suffixes, e.g.
  `"श्रम ऐन, २०७४"` — any year-stripping regex must match `[०-९]+`, not
  ASCII `\d+`. (`retrieve_postgres()` already digit-folds the *query* via
  `_preprocess()`/`_DIGIT_MAP`, but `title_ne` in the DB is never
  digit-folded — don't fold it, match Devanagari digits in the title
  directly.)
- **Design question already resolved with Prakash** (AskUserQuestion):
  alias data source is a **new small static file**, not a DB table —
  same "tooling, not data" split as AGENT-34's golden sets. Seed it thin
  (a handful of obvious cases), Prakash expands it over time. No dual-
  approval needed (this is a matching-convenience layer, not authority
  data — PS-2/PS-5 don't apply here).

## Locked design

All changes in `app/retrieval/postgres_retriever.py` unless noted.

### 1. Section ranges

New `_parse_section_range(query: str) -> tuple[int, int] | None`, tried
**before** `_parse_section_reference()` in `retrieve_postgres()`:
`re.search(r"(?:दफा|धारा)\s*(\d+)\s*(?:देखि|-|–|—)\s*(\d+)\s*(?:सम्म)?", query)`
(query is already digit-folded via `_preprocess()` by the time this runs,
so ASCII `\d+` is correct here — unlike the title-matching case above).
If it matches, use the range and **do not** also call
`_parse_section_reference()` for this query (avoid a conflicting single-
number filter alongside a range filter). If it returns `None`, fall back
to the existing single-number parse exactly as today — no behavior change
for non-range queries.

`exact_lookup_search()`: when a range is present, replace the section
equality filter with:
```sql
(
    NULLIF(regexp_replace(c.section_number, '\D', '', 'g'), '')::int
        BETWEEN %(low)s AND %(high)s
    OR NULLIF(regexp_replace(c.parent_section, '\D', '', 'g'), '')::int
        BETWEEN %(low)s AND %(high)s
)
```
(defensive regex-strip before casting — safe no-op on today's pure-digit
data, tolerant if a non-digit character ever appears rather than raising).
When no range is present, keep the existing single-number equality filter
byte-for-byte unchanged.

### 2. Year-suffix matching fix (independent of aliases)

`_resolve_act_titles()` (renamed, see #3) must match a query against
`title_ne` **with its trailing year clause optionally stripped**, not
only the full literal title. Add a second match condition alongside the
existing full-title `strpos`:
```sql
strpos(%(query)s, title_ne) > 0
OR strpos(%(query)s, regexp_replace(title_ne, ',\s*[०-९]+\s*$', '')) > 0
```
Keep the full-title branch (don't remove it — a query that *does* include
the year should still match, and ranking/ordering should still prefer the
more specific full match — see ORDER BY below).

### 3. Act aliases + multiple Acts per query

New file: **`app/retrieval/act_aliases.json`** — `{"<colloquial or
alternate Nepali name>": "<canonical title_ne text to search for, minus
year if you want the year-optional match to apply>"}`. Seed with a
small number of real, verifiable examples only (no guessing at names you
haven't confirmed against actual `work.title_ne` rows) — if you can't
verify a colloquial name is real and correct, leave it out rather than
invent one (`AGENTS.md`'s "jagged intern" — don't fabricate legal-content
data).

Load once at module level (mirror how other static config in this module
is loaded — keep it simple, a plain `json.load` at import time, no
caching abstraction beyond that).

Rename `_resolve_act_title` → `_resolve_act_titles(conn, query) ->
list[str]` (work_id list, not a single string):
- Build an augmented search text: the original query, plus — for every
  alias key found as a substring of the query — the alias's mapped
  canonical text appended (so alias hits flow through the *same*
  `strpos`-based SQL matching as #2, no separate code path to keep in
  sync).
- Drop `LIMIT 1`; instead `ORDER BY length(title_ne) DESC LIMIT 5` (bound
  the candidate count — a small, explicit cap, not unlimited) and return
  all matched `id`s as a list.
- Known, explicitly out-of-scope simplification (state this in code
  comments, don't silently hide it): if a query names two Acts each with
  their own दफा number (e.g. "कम्पनी ऐन दफा ५ र श्रम ऐन दफा १०"), this task
  does **not** pair section numbers to specific Acts — any parsed section
  number/range applies as a global filter across *all* matched Acts'
  `work_id`s. Per-Act section pairing is a materially harder parsing
  problem and out of scope.

`exact_lookup_search()`: `c.work_id = ANY(%(work_ids)s)` when the list is
non-empty (replaces the single `=` equality filter).

### 4. Provisos referenced by name

New `_parse_proviso_reference(query: str) -> bool`:
`bool(re.search(r"परन्तुक|स्पष्टीकरण", query))`.

`exact_lookup_search()`: when `True` **and** a section number or range is
also present, AND an additional `c.level = 'proviso'` clause onto the
section/range filter (narrows to just the proviso chunk of that दफा,
rather than the undifferentiated `section_number OR parent_section` match
that returns operative + subsections + proviso together today). When the
proviso keyword is present but no दफा number/range is given, do nothing
special — there's nothing to anchor the filter to; let vector/lexical
handle it as today.

## Explicitly forbidden

- Do not touch `app/ingestion/laws_chunker.py`, `pgvector_indexer.py`, or
  any ingestion path — the chunker's digit-only दफा parsing and the
  `level`/`section_number`/`parent_section` write paths are out of scope,
  even if you notice something about them while reading. Report, don't
  fix.
- Do not touch `_parse_subsection_reference` — leave it exactly as the
  known, already-reviewed dead code it is.
- Do not touch the RRF fusion, relevance threshold, rerank call, or the
  vector/lexical search branches.
- Do not touch `eligibility_gate.py`, `validation_gate.py`, or any
  temporal/authority-store code.
- Do not build a fuzzy-matching/typo-tolerance layer — this task is exact
  structural + alias matching only, not approximate matching.
- Do not populate `act_aliases.json` with names you can't verify against
  actual `work.title_ne` data — a small, correct seed beats a large,
  guessed one.

## Must verify against live DB before relying on it (Pi has live DB
access this session; I don't — say so honestly in your completion report
if any of these turn out different from what's assumed above, don't
silently code around a surprise)

1. Spot-check several `chunks.section_number`/`parent_section` values —
   confirm they're pure digit strings (no letter suffixes like "५क") as
   the chunker regex implies. If letter-suffixed values exist in the live
   corpus despite the regex, the range filter's digit-strip-and-cast is
   still safe (defensive `NULLIF`/`regexp_replace`), but flag it rather
   than assuming it's fine.
2. Spot-check several `chunks.level` values for law chunks — confirm
   `'proviso'` actually appears (not just `'section'`/`'subsection'`) and
   that `chunk_type` does **not** already encode this distinction (i.e.
   confirm the `level` column, not `chunk_type`, is the right filter
   column).
3. Spot-check several `work.title_ne` values — confirm the trailing
   `, <Devanagari-digit year>` pattern is consistent enough for the
   year-stripping regex to be worth adding (not a rare edge case).

## Required tests

Extend `tests/test_retrieval.py`'s existing `Cursor`/dispatch-on-SQL-
substring mock pattern (per AGENT-30's own precedent — don't build a
parallel mock). Cover:

1. `_parse_section_range` matches `"दफा ५ देखि १० सम्म"` (post-fold: "5"
   through "10") and a hyphenated form; returns `None` for a plain single-
   number reference (no regression on the existing single-number path).
2. A range query produces a `BETWEEN`-shaped SQL filter, not the
   equality filter, and a plain single-number query still produces the
   original equality filter unchanged.
3. `_resolve_act_titles` matches a year-less title (`"श्रम ऐन"` matching
   `"श्रम ऐन, २०७४"`) — the fix from #2, independent of aliases.
4. `_resolve_act_titles` matches via an alias-file entry that a raw
   `title_ne` substring search would miss on its own.
5. A query naming two different Acts (real or seeded-alias examples)
   returns both `work_id`s, and `exact_lookup_search`'s filter uses
   `= ANY(...)`, not `=`.
6. `_parse_proviso_reference` detects both परन्तुक and स्पष्टीकरण; a
   proviso-keyword query combined with a दफा number adds the
   `level = 'proviso'` clause; a proviso-keyword query with **no** दफा
   number does not change the filter at all (falls through unchanged).
7. Full round-trip through `retrieve_postgres()` (not just the helper
   functions in isolation) for at least one range case and one multi-Act
   case, proving the SQL actually gets built and executed via the mocked
   cursor — matching the "prove the wiring, not just the unit" standard
   this whole program has used since AGENT-29/31.

## Required checks before reporting done

`make test`, `make lint`. `scripts/label_eval_candidates.py` is not
covered by the Makefile's fixed lint file list (a pre-existing gap noted
several times this session) — `postgres_retriever.py` **is** already in
that list, so ordinary `make lint` covers this task's changed file; no
extra manual lint step needed here. `make eval-gates` is not relevant
(no gate/serve-path code changes).

## Commit authorship

Every commit on this branch must be authored as
`Prakash Basnet <basnetprakash090@gmail.com>` — no AI/Claude attribution,
no `Co-Authored-By` trailer.
