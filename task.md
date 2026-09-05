# AGENT-39: dual-query normalization for English/Roman/hybrid legal queries

## Objective

Retrieval already runs the original query *and* a Nepali-translated
variant through vector and lexical search — that part is not broken.
The actual gap is narrower: (1) the translation trigger skips some
genuinely hybrid queries because it thresholds on character-majority
rather than presence of untranslated content, and (2) exact
legal-reference parsing (section/range/schedule/proviso/act-title) only
ever looks at the original query, so it misses references written in
the translated variant. Fix both, without changing gates, validation,
RRF, rerank, or answer generation.

Prakash supplied an initial spec for this. Grounding below confirms
which parts of it match the real code, and corrects two things that
don't: a SQL construct in its schedule-lookup suggestion that isn't
valid Postgres, and a function (`_is_devanagari`) it proposed replacing
that's actually depended on by a stress test outside this task's scope.

## Grounding (already done, don't re-derive)

Read `app/retrieval/postgres_retriever.py` and `tests/test_retrieval.py`
in full, plus `tests/stress/test_romanized_eligibility.py`, before
writing this. Current state, precisely:

- `retrieve_postgres()` (`postgres_retriever.py:215-488`) already does
  **dual vector + dual lexical** search: `qvec`/`qvec_ne` both go
  through `vector_search()`, `query`/`query_ne` both go through
  `lexical_search()`, and all four result sets feed the same RRF fusion
  (`ranked_lists`, lines 402-408). This part of "use both variants for
  vector + lexical search" is **already done** — don't rebuild it, don't
  touch it beyond what's needed for the exact-parser fix below.
- The actual gap is in the **exact-lookup path**: `section_range`,
  `section_nums`/`section_num`, `schedule_num`, `proviso_ref` are all
  parsed from `query` alone, at lines 220-228 — and critically, *before*
  `query_ne = translate_query(query)` is even computed (line 230). So an
  English/Roman query whose section reference only exists in the
  translated Nepali form (e.g. "Labor Act section 5" → translated
  "श्रम ऐन दफा ५") never gets its exact-lookup filters populated, because
  `दफा|धारा` (the only patterns `_parse_section_numbers`/`_parse_section_range`
  look for) never appear in the English original. This is the real bug
  to fix.
- `_resolve_act_titles(conn, query)` (lines 165-186) has the same
  problem via a different path: it's called on `title_query` (line 264
  — `unicodedata.normalize("NFC", query)`, the *original pre-`_preprocess`*
  query) only, never on a translated variant.
- **`title_query` intentionally skips digit-folding** — `_preprocess()`
  (line 73-74) NFC-normalizes *and* folds Devanagari digits (०-९) to
  ASCII, but `title_query` (line 218) only NFC-normalizes, computed
  before `query` gets reassigned to the folded form. This is
  deliberate: `work.title_ne` values in the DB contain Nepali-numeral
  years (e.g. `"श्रम ऐन, २०७४"`), so `strpos(%(query)s, title_ne)`
  (line 178) needs the query to still contain the *original* Devanagari
  digits to match. Folding digits before title-matching would silently
  break matching against titles with Nepali-numeral years. **Whatever
  variant is passed to `_resolve_act_titles` must go through the same
  NFC-only treatment `title_query` gets today — never the digit-folded
  `_preprocess()`'d form.**
- **`translate_query()`'s current trigger is `_is_devanagari(query)`**
  (line 83, defined lines 77-79: `>50%` of characters in the Devanagari
  Unicode block → skip translation). This is the actual reason "hybrid
  queries can skip translation" — a query that's, say, 60% Devanagari
  characters by count but contains a meaningful untranslated English
  Act name or phrase still gets skipped. Prakash's proposed replacement
  — trigger translation whenever *any* ASCII letter is present — fixes
  this correctly and was checked against all three existing
  `test_translate_query_*` tests (lines 255-300 in
  `tests/test_retrieval.py`): all three still pass unchanged with the
  new trigger, because they test either pure-Nepali input (still
  correctly skips) or cases where translation is independently blocked
  by a missing API key / API failure (unaffected by the trigger
  change).
- **`_is_devanagari` itself must NOT be removed, renamed, or have its
  behavior changed.** It has 3 direct unit tests in
  `tests/test_retrieval.py` (lines 165-174) *and* is called directly by
  `tests/stress/test_romanized_eligibility.py:27`
  (`r._is_devanagari("muluki ain ko dafa ek")`) — a file **outside this
  task's stated scope** (only `postgres_retriever.py` and
  `tests/test_retrieval.py` are in scope). Add the new trigger as a
  separate function; leave `_is_devanagari` exactly as it is, dead code
  or not.
- **`tests/stress/test_romanized_eligibility.py` also imports `Conn`,
  `Cursor`, `ROW1`, `patch_common` from `tests.test_retrieval`**
  (its line 8). Whatever you change in `tests/test_retrieval.py`'s test
  helpers, those four names must keep working with compatible
  signatures for that external file — don't rename or restructure them,
  extend if you need to.
- **The one existing SQL construct in Prakash's spec that won't run:**
  `strpos(c.chunk_text, 'अनुसूची ' || ANY(...))` is not valid Postgres —
  `ANY(...)` can only appear on the right-hand side of a scalar
  comparison (`x = ANY(array)`), not spliced into a string
  concatenation. For multiple schedule numbers, use:
  ```sql
  EXISTS (
      SELECT 1 FROM unnest(%(schedule_nums)s) AS sn
      WHERE strpos(c.chunk_text, 'अनुसूची ' || sn) > 0
  )
  ```
  This is the "boring" option the spec itself allows for when the
  first suggestion is awkward — use it, don't invent a third approach.
- **Eligibility is already enforced correctly and must stay that way**:
  every SQL query in `exact_lookup_search()` starts its `filters` list
  with `"c.id::text = ANY(%(eligible)s)"` (line 304). Any new filter
  clause (multi-schedule, variant-merged section nums, etc.) must be
  *appended* to that list, never replace or route around it.
- Existing tests that call `retrieve_postgres()` through `patch_common`
  (which monkeypatches `translate_query` to always return `None` —
  `tests/test_retrieval.py:97-101`) exercise the exact-lookup path with
  **no translated variant at all**. That means: with `query_ne is None`,
  every one of those existing tests
  (`test_exact_lookup_only_result_survives`,
  `test_exact_lookup_uses_work_title_filter`,
  `test_range_query_uses_between_not_equality`,
  `test_multiple_sections_use_any_filter`,
  `test_retrieve_multi_act_query_uses_all_work_ids`,
  `test_proviso_filter_requires_section_anchor`,
  `test_huge_range_does_not_fallback_to_single_section`,
  `test_plain_section_keeps_equality_filter`,
  `test_bare_subsection_does_not_run_exact_section_filter`) must
  produce **byte-identical SQL/params** to today. That's your regression
  safety net for the "merge across variants" logic — when there's only
  one variant, merging must be a no-op.

## What to build

### 1. Translation trigger

Add, near `_is_devanagari`:

```py
def _needs_nepali_variant(query: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in query)
```

Change `translate_query()`'s guard from `if _is_devanagari(query): return
None` to `if not _needs_nepali_variant(query): return None`. Nothing
else in `translate_query()` changes. Do not touch `_is_devanagari`.

### 2. Reorder + variant construction in `retrieve_postgres()`

Move the `query_ne = translate_query(query)` / `_preprocess(query_ne)`
block (currently lines 230-232) to run *before* the section/schedule/
proviso parsing block (currently lines 220-228), so parsing can see the
translated variant. Also capture the pre-fold translated text for title
matching (see §4):

```py
translated_raw = translate_query(query)          # query already _preprocess()'d
title_query_ne = (
    unicodedata.normalize("NFC", translated_raw) if translated_raw else None
)
query_ne = _preprocess(translated_raw) if translated_raw else None

query_variants = [query]
if query_ne and query_ne != query:
    query_variants.append(query_ne)
```

Everything downstream that already uses `query_ne` for vector/lexical
search (lines 265-266, 348-349, 366-367, 403-406) keeps working
unchanged — you're only moving *when* it's computed, not what it's used
for there.

### 3. Exact parser over both variants

Replace the single-variant parsing block with variant-aware versions.
Shape (adapt to actual style, this is the logic not literal code):

```py
def _first_match(fn, variants):
    for v in variants:
        result = fn(v)
        if result:
            return result
    return None

section_range = _first_match(_parse_section_range, query_variants)
if section_range or any(_SECTION_RANGE_RE.search(v) for v in query_variants):
    section_nums = []
else:
    seen = []
    for v in query_variants:
        for n in _parse_section_numbers(v):
            if n not in seen:
                seen.append(n)
    section_nums = seen[:_MAX_SECTION_NUMBERS]
section_num = section_nums[0] if len(section_nums) == 1 else None
schedule_nums = []
for v in query_variants:
    for n in _parse_schedule_numbers(v):
        if n not in seen_schedule:  # same dedupe-preserve-order pattern
            ...
proviso_ref = any(_parse_proviso_reference(v) for v in query_variants)
```

Keep `_MAX_SECTION_RANGE`/`_MAX_SECTION_NUMBERS` caps applied exactly as
today (range check inside `_parse_section_range` per-variant already
caps; number-list cap applies *after* the cross-variant merge, same
final cap value). No fuzzy matching, no new heuristics beyond "check
each variant with the existing exact parsers and merge."

### 4. `_parse_schedule_reference` → `_parse_schedule_numbers`

Mirror the existing `_parse_section_reference`/`_parse_section_numbers`
relationship (lines 130-140): add
`_parse_schedule_numbers(query: str) -> list[str]` that finds *all*
`अनुसूची\s*(\d+)` matches (not just the first), then make
`_parse_schedule_reference` a thin wrapper: `nums =
_parse_schedule_numbers(query); return nums[0] if nums else None`. This
keeps `_parse_schedule_reference`'s existing test
(`test_parse_schedule_reference`) passing unchanged. Merge
`schedule_nums` across variants the same dedupe-preserve-order way as
section numbers, no separate cap needed unless you judge one useful
(keep it boring).

In `exact_lookup_search()`, replace the single-schedule filter
(currently `schedule_num is not None` → one `strpos(...) > 0` clause)
with the `schedule_nums` list version using the `EXISTS (SELECT 1 FROM
unnest(...) ...)` form from the grounding section above. If
`schedule_nums` has exactly one entry this must produce the same
practical filtering result as today's single-schedule path (existing
behavior preserved, just expressed generally).

### 5. Act-title lookup over both variants

```py
title_variants = [title_query] + ([title_query_ne] if title_query_ne else [])
act_work_ids = []
for tv in title_variants:
    for wid in _resolve_act_titles(conn, tv):
        if wid not in act_work_ids:
            act_work_ids.append(wid)
act_work_ids = act_work_ids[:5]
```

Do not change `_resolve_act_titles()`'s signature or internal SQL — it
has 4 direct unit tests pinned to its current single-query signature
(`test_resolve_act_titles_longest_match_wins` and 3 others). Call it
once per title variant from `retrieve_postgres()` instead.

## Explicitly forbidden

- `eligibility_gate.py`, `validation_gate.py`, `query_graph.py`,
  prompts, ingestion, eval gates — do not touch, per the original spec.
- Any change to `_is_devanagari`'s name, signature, or behavior.
- Any change to `_resolve_act_titles()`'s signature.
- Any change to `retrieve_postgres()`'s external signature —
  `gated_orchestrator.py`, four `app/eval/*_slice.py` files,
  `scripts/query.py`, and `scripts/label_eval_candidates.py` all call it
  positionally; it must keep working unmodified for all of them.
- Digit-folding the query before it reaches `_resolve_act_titles` (see
  the `title_query`/Nepali-numeral-years grounding above) — this would
  be a silent regression, not caught by any existing assertion, so
  don't introduce it.
- Any SQL construct that doesn't actually run on Postgres — if unsure,
  favor the explicit `EXISTS (SELECT ... FROM unnest(...))` pattern over
  anything clever with `ANY()` inside a string expression.
- Fuzzy/approximate matching anywhere in the exact-lookup path.

## Required checks

- `make test && make lint` — must stay fully green.
- Every existing test in `tests/test_retrieval.py` and
  `tests/stress/test_romanized_eligibility.py` must keep passing
  **unmodified** wherever it exercises the `query_ne is None` path
  (that's the regression guarantee described in the grounding section).
  Tests that legitimately need updating because they directly assert
  the old single-variant call shape may be updated — use judgment, but
  the bar is "byte-identical SQL/params when there's no translated
  variant," not "same source code."
- New/extended tests in `tests/test_retrieval.py` per Prakash's original
  list:
  - pure Nepali does not call Gemini translation (extend, don't
    duplicate, the existing `test_translate_query_skips_devanagari`)
  - pure English calls translation
  - Roman Nepali calls translation
  - hybrid ASCII + Nepali calls translation
  - exact parser uses the translated variant: original `"Labor Act
    section 5"`, mocked translation `"श्रम ऐन दफा ५"`, assert the
    resulting exact-lookup SQL/params include both the Act's work_id
    filter and the section-number filter
  - original query is still embedded/searched (already true — assert
    it stays true)
  - translated query is also embedded/searched (already true — assert
    it stays true)
  - translation returning `None` still retrieves correctly (already
    covered by `patch_common`-based tests — confirm, don't skip)
  - a translated variant identical to the original is not searched
    twice (already handled by the `query_ne != query` check — add a
    test if one doesn't already cover it)

## Commit authorship

Every commit authored `Prakash Basnet <basnetprakash090@gmail.com>` — no
AI attribution, no `Co-Authored-By: Claude` trailer.

## Branch

`agent/dual-query-retrieval`, off `dev`.
