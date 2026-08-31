# AGENT-18 — Correlate `<amend>` tags with the document's amendment table into `lifecycle_effect`

**Branch:** `agent/amend-tag-correlation`
**Base:** `dev`
**Engineer:** Pi

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No AI attribution of any kind, ever.

---

## Objective

Every दफा/नियम in this corpus that has been amended carries an inline
`<amend>...</amend>` marker (stripped before the model ever sees it —
`parser.py::_clean_text()`, PS-12 unaffected, don't touch that). Today
that annotation is discarded entirely; the authority store has **zero**
`amend`-type `lifecycle_effect` facts from real corpus data (mirrors
AGENT-16's starting point for `repeal` before that task — confirmed the
same way: `effect_type` already allows `'amend'` in migration 001's CHECK
constraint, and `EffectType.AMEND` already exists in `app/authority/
models.py` — **no schema change needed**, pure extend-existing per
Ponytail).

Extract, per दफा component, which act/regulation amended it and when, as
pending `amend` lifecycle facts through the existing dual-approval path.

## Grounding already done (re-verify counts, don't re-derive from scratch)

**This corrects the old backlog note's characterization — re-verify before
trusting either version.** The note carried forward from AGENT-16's
scoping said "83% dominant [named-act] pattern, ordinal position for a
real secondary pattern." A precise classification of all 15,748 tags
across all 677 `laws.jsonl` records (not just a rough keyword sweep) found
the *opposite* emphasis:

```
named-act pattern:      4,926 / 15,748  (31.3%)
ordinal-position:        8,808 / 15,748  (55.9%)  ← actually dominant, not secondary
gazette-date-only:          24 / 15,748  ( 0.2%)
other/heterogeneous:      1,990 / 15,748  (12.6%)  ← NOT amendment records, see below
```

Reproduce with a regex classifying each `<amend>...</amend>` tag's inner
text: `ऐन,?\s*[०-९]{4}\s*द्वारा` = named-act; a leading Devanagari ordinal
word (पहिलो/दोस्रो/... — see reuse note below) immediately before
"संशोधन" = ordinal-position; a leading BS date = gazette-date-only;
anything else = other.

1. **Named-act pattern (31.3%)** — tag content is literally
   `<Act/Regulation name>, <BS year> द्वारा {संशोधित|थप}।`. E.g.
   (`सुशासन_(व्यवस्थापन_तथा_सञ्चालन)_ऐन_२०६४`):
   `<amend>केही नेपाल ऐन संशोधन गर्ने ऐन, २०७२ द्वारा संशोधित।</amend>`.
   Resolve by normalized-title match — same normalization
   `enabling_extractor.py::_normalize_title()` already does (comma→space,
   NFC) — against the entries in this document's own amendment table (see
   below), **not** a `work`-table lookup like
   `enabling_extractor._resolve_work()`: many amending acts named in these
   tags aren't necessarily in the `work` table as their own row, but they
   are always listed in the amended document's own table.

2. **Ordinal-position pattern (55.9%, the actual majority)** — tag content
   is `<ordinal word> संशोधनद्वारा {संशोधित|थप}।`, e.g. `<amend>पहिलो
   संशोधनद्वारा संशोधित।</amend>` — references the table by row position,
   not by name. Ordinal words found in this corpus (पहिलो through
   पन्ध्रौं, i.e. 1st–15th) are a strict subset of
   `app/ingestion/commencement_extractor.py::ORDINAL_DAYS` — **reuse that
   dict directly** (same word→number mapping; AGENT-12 uses the number as
   a day-offset, this task uses it as a 1-based table row position) rather
   than redefining a second ordinal-word table. Safe-abstain (don't guess)
   on any ordinal word not in that table, same discipline AGENT-12 used.

3. **Gazette-date-only (0.2%, 24 instances)** — tag is just a BS date, no
   act name or ordinal, e.g. structurally mirrors AGENT-12's
   gazette-dependent commencement shape. Thin evidence at this volume —
   **abstain, don't build special handling for it**, same call AGENT-16
   made on comparably-thin patterns.

4. **Other (12.6%, 1,990 instances) — explicitly out of scope, do not
   attempt.** This bucket is *not* a harder amendment sub-pattern, it's
   mostly not amendment records at all: commencement dates already
   covered by AGENT-12 (`"यो ऐन ... देखि लागू भएको"`), repeal asides
   already covered by AGENT-16 (`"यो ऐन हाल खारेज भएको"`), name-change
   notes, unrelated constitutional cross-references, executive
   pay-adjustment decisions, and a small number of tags that combine an
   ordinal *and* an act name in one clause (nested/ambiguous — don't
   guess which one wins). Attempting to classify all of this risks
   producing duplicate or conflicting `lifecycle_effect` rows against
   what AGENT-12/16's dedicated extractors already write from the
   *primary* clause elsewhere in the document. **Any `<amend>` tag that
   doesn't cleanly match pattern 1 or 2 above must be skipped, not
   guessed at** — count and report skips, don't silently drop without
   accounting for them.

### The amendment table itself (new parsing surface, not done anywhere today)

Every amended document has an ordered table near the top, before the
substantive text, that both resolution patterns above depend on:

- **Acts**: headed `संशोधन गर्ने ऐन`, e.g.
  `१. केही नेपाल ऐन संशोधन गर्ने ऐन, २०७२    २०७२।११।१३`
- **Regulations**: headed bare `संशोधन` or `संशोधन गर्ने नियम` (different
  label, same shape — regulations are amended by regulations, not acts).

Corpus-wide: 309/519 amend-tag-bearing documents have the `संशोधन गर्ने
ऐन` heading; +187 more (496/519, 96%) are covered once the boundary
regex also accepts the bare/regulation heading variant. **23/519 (4%)
have no discoverable table at all — abstain for these, don't fabricate a
correlation.** Each table row is `N. <name>, <BS year>    <BS date>` —
the trailing date is the same BS-date shape already handled everywhere
else in this codebase (`app/authority/bs_ad_calendar.py::lookup()`,
already used throughout `parser.py`) — reuse it for `effective_date`
resolution, don't write a new date parser.

### Enclosing दफा

The `<amend>` tag's character offset in the raw `content` string falls
inside exactly one दफा component's span, using the same offset
boundaries `parser.py::parse_law()` already computes between consecutive
`_HEADER_RE` matches — this extractor needs those spans (or to recompute
equivalent ones), not the already-stripped `component.text_ne` (tags are
removed by `_clean_text()` before that field is populated — this is
correct existing behavior per PS-12, don't change it). Same pattern
`commencement_extractor.py`/`repeal_extractor.py` already use: they scan
raw `content` directly, not the parsed/cleaned components.

## Required fix

1. **`app/ingestion/amend_extractor.py`** (new, deterministic, no LLM —
   same shape as `commencement_extractor.py`/`repeal_extractor.py`):
   - Parse the document's amendment table into ordered
     `(position, normalized_name, bs_date)` entries.
   - For each `<amend>...</amend>` tag: classify (named-act / ordinal /
     abstain-everything-else per the grounding above), resolve to a table
     entry, resolve `effective_date` via `bs_ad_calendar.lookup()`,
     resolve the enclosing दफा component URI via offset matching.
   - Return one proposal per resolved tag (dedupe reasonably — the same
     दफा is very often amended multiple times by different acts, each is
     a distinct fact, don't collapse them; but the same tag content
     repeated verbatim for the *same* दफा *and* same amending act is
     presumably one real edit split across what the parser sees as
     nearby matches — use judgment, verify against a few real documents
     by hand before deciding).
2. **`app/authority/writer.py::propose_lifecycle_amend()`** (new) —
   mirrors `propose_lifecycle_commence()`'s shape: always
   `approval_status='pending'`; `legal_valid_time` from the resolved
   `effective_date` when resolvable, `'empty'` + a documented
   `commencement_dependency`-style sentinel when not (no fabricated
   date — PS-2's discipline applies here too even though this isn't a
   commencement fact); dedup on
   `(component_uri, effect_type='amend', approval_status='pending')` or a
   more specific key if you find same-दफा-different-amendment collisions
   during testing — your call, document it.
3. **`app/ingestion/pipeline.py`** — wire into the existing
   `PROPOSE_LIFECYCLE` span (AGENT-12, extended by AGENT-16 for repeal),
   same `SAVEPOINT`/`ROLLBACK TO SAVEPOINT` best-effort discipline. Not a
   new pipeline stage.
4. **`scripts/review_lifecycle.py`** — already generalized to accept any
   `--effect-type` (AGENT-16). Verify `amend` proposals are listable/
   approvable through it as-is; only touch this file if you find it
   doesn't actually work for `amend` (report why, don't just patch
   blindly).

## Explicit non-goals (do not attempt)

- **No pre-amendment text reconstruction.** This corpus is a single
  current-snapshot — there is no "before" text to version. An `amend`
  `lifecycle_effect` row records the *fact* (this दफा was amended by X on
  date Y), not a text diff. This means today's output does **not**
  achieve full PS-17 compliance (closing a prior `expression`'s
  `valid_time` at the retroactive date) — there's no prior `expression`
  row to close. State this plainly in your return writeup as an honest
  scope note, same as AGENT-11/12/13/16's "doesn't wire into retrieval
  yet" caveats — don't paper over it.
- **No handling of the "other" 12.6% bucket** — see grounding above.
- **No change to `parser.py`'s `_AMEND_RE`/`_clean_text()` stripping** —
  still correct, PS-12 unaffected.
- **No schema/migration change** — `'amend'` is already an allowed
  `effect_type`.

## Scope / allowed files
- `app/ingestion/amend_extractor.py` (new)
- `app/authority/writer.py` (add `propose_lifecycle_amend`, don't touch
  the existing `commence`/`repeal` functions)
- `app/ingestion/pipeline.py` (wire into `PROPOSE_LIFECYCLE` only)
- `scripts/review_lifecycle.py` (only if you find a real gap, see above)
- `tests/test_amend_extractor.py` (new, covers `writer.py::
  propose_lifecycle_amend` too — `tests/test_commencement_extractor.py`
  and `tests/test_repeal_extractor.py` both test their writer function in
  the same file as the extractor, no separate `test_authority_writer.py`
  exists in this repo, follow that convention), relevant additions to
  `tests/test_ingestion_pipeline.py` for the `PROPOSE_LIFECYCLE` wiring

**Forbidden:** `app/authority/parser.py`, `eligibility_gate.py`,
`validation_gate.py`, any retrieval-path file (out of scope, unrelated);
`commencement_extractor.py` / `repeal_extractor.py` (different subsystem,
already correct — reuse their constants/patterns, don't modify them);
anything in the "other" 12.6% bucket.

## Required checks
- `make test`
- `make lint`
- `make eval-gates` — zero-tolerance gates
  (`repealed-as-current`, `not-yet-effective-as-current`,
  `overruled-as-good-law`) must stay at 0
- Corpus-wide count: how many `amend` proposals generated, broken down by
  resolution method (named-act / ordinal), how many tags skipped and why
  (should roughly match the classification counts above — report the
  actual numbers, they may differ once you're working with real spans
  instead of a standalone regex sweep)
- Spot-check by hand (per AGENTS.md "become one with the data"): pick 3-5
  real documents, verify the resolved `(दफा, amending act, date)` triples
  against the raw text yourself before trusting the aggregate count

## Zero-tolerance gates guarded
`repealed-as-current = 0`, `not-yet-effective-as-current = 0` — this task
adds a new lifecycle_effect writer path; confirm neither gate regresses.

## Self-review checklist before pushing
- No path skips the eligibility gate (this task doesn't touch it — confirm
  you didn't add one)
- Model never writes citations (unaffected — ingestion-side only)
- No new schema/table/column, no new abstraction beyond
  `propose_lifecycle_amend`/`amend_extractor.py` (Ponytail — both are
  extend-existing, not new mechanisms)
- Every `amend` proposal written is `pending`, never auto-approved
- No fabricated `effective_date` — empty range + honest gap when
  unresolvable, matching PS-2/AGENT-12's discipline

## Return
Commit hash(es), changed files, checks run/results, the corpus-wide
proposal count broken down by resolution method, the hand-verified
spot-check documents and what you found, the honest PS-17 scope note,
assumptions made, and any remaining risks.
