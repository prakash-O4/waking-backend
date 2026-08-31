# AGENT-16 — Extract whole-act repeal declarations into lifecycle_effect

## Objective

When a new ऐन supersedes an older one, it almost always says so explicitly
in its own text — a "खारेजी र बचाउ" (Repeal and Saving) दफा, or an
equivalent standalone clause, naming the repealed act and stating it is
repealed, followed by savings sub-clauses (prior acts done under the old
act remain valid, pending matters continue under the new act, etc.).
This is the mechanism the `repealed-as-current = 0` zero-tolerance gate
exists for, and today **the authority store has zero repeal facts** —
`lifecycle_effect` has no `effect_type='repeal'` rows from real corpus
data at all (confirmed: AGENT-13's backfill only ever wrote `commence`
effects).

### Corpus grounding (script-verified against all 677 `laws.jsonl` records)

A precise operative-clause regex — the named act/ordinance/regulation
(`... ऐन|अध्यादेश|नियमावली|नियम, <BS year>`) sitting **directly** before
the phrase `खारेज गरिएको छ`, with nothing (no `को दफा`, `को उपदफा`, `वाक्यांश`)
in between — finds **157 documents, 157 clean whole-act repeal
declarations**, one per document. Example (`लेखापरीक्षण_ऐन_२०७५`):

```
३०. खारेजी र बचाउ : (१) लेखापरीक्षण ऐन, २०४८ खारेज गरिएको छ ।
(२) लेखापरीक्षण ऐन, २०४८ बमोजिम भए गरेका काम कारबाही यसै ऐन बमोजिम भए गरेको मानिनेछ ।
(३) यो ऐन प्रारम्भ हुँदाका बखत ... यसै ऐन बमोजिम हुनेछ ।
```

Sub-clause (१) is the operative repeal; (२)/(३) are savings provisions —
note they do **not** repeat the phrase `खारेज गरिएको छ`, so the strict
regex naturally skips them without extra logic. Verify this holds broadly
before relying on it structurally.

A **looser** version of the same phrase (`खारेज गरिएको छ` anywhere, not
requiring a clean act-name-then-verb boundary) matches 219 times — the
other 62 are a **structurally different, harder pattern**: partial
repeal of a *specific दफा of another act* (e.g. `"सवारी तथा यातायात
व्यवस्था ऐन, २०४९ को दफा १६८ को प्रतिबन्धात्मक वाक्यांश खारेज गरिएको छ"`)
or list-style consequential-amendment दफाs mixing repeal of one act's
दफा with text-substitution edits to another's (e.g.
`प्रशासकीय_अदालत_ऐन_२०७६`'s दफा listing `(क)` repeal-of-दफा,
`(ख)`/`(ग)` "...शब्दहरूको सट्टा ... शब्दहरू राखिएका छन्" substitutions).
**Do not attempt these in this task** — see Out of scope.

### Where this writes, and why no migration is needed

`lifecycle_effect.effect_type` already has `'repeal'` in its CHECK
constraint (migration 001) and the `is_eligible()` SQL function already
excludes a component when an approved `repeal`/`expiry`/`declared_invalid`
effect's `legal_valid_time` lower bound is `<= as_of` (`migrations/
001_bitemporal_schema.sql:86-92`). **Nothing in the schema needs to
change.** But `lifecycle_effect` is keyed by `component_uri` (a single
दफा/धारा/परिच्छेद/full-document component), not by `work.uri` — so
"repeal this whole act" means writing one `repeal` proposal **per
component belonging to the repealed work** (`SELECT uri FROM component
WHERE uri LIKE '{work.uri}/%'` — this also naturally covers documents
with no दफा structure, whose only component is `.../full/1`).

### The repeal date is not locally known — do not fabricate it (PS-2)

A repeal's effective date is normally the *repealing* act's own
commencement date, not something stated in the repeal clause itself.
That commencement date may itself be unresolved (gazette-notification-
pending, per AGENT-12). `pipeline.py::_commence_date()` (line ~803)
shows the existing per-दफा commence lookup pattern and the
`commencement_dependency` column exists precisely for "known to depend
on X, not yet resolved" honesty (see AGENT-12's `no_commencement_clause`/
`gazette_notification_pending` sentinels). Design the repeal proposal's
dependency/date handling the same way — a repeal proposal for an
approved-but-undated dependency is a legitimate, honest state; a
fabricated date is not. State your design and why in your return
message; this brief does not prescribe the exact resolution mechanism.

### `scripts/review_lifecycle.py` is hardcoded to `commence` — it will not surface repeal proposals as-is

`--list`/`--list-work` filter on `effect_type='commence'` explicitly
(`scripts/review_lifecycle.py:35,152`), and `propose_lifecycle_commence`'s
dedup query is similarly commence-specific. A repeal proposal that the
CLI can never list is a proposal that can never be approved — silently
defeating Core Invariant #5 (human-gated, dual approval) even though the
row technically stays `pending` forever. This needs a minimal fix (e.g.
an `--effect-type` flag, default showing all types) — extend, don't
replace.

### Honest scope note — carry this forward, it is still true after this task

This task populates the authority store with real repeal facts. It does
**not** make retrieval respect them: `app/retrieval/eligibility_gate
.py::eligible_chunk_ids()` (the function retrieval actually calls) derives
eligibility from `documents.ingestion_status`/`chunks.effective_date_ad`
only — it does not query `lifecycle_effect` at all. The `is_eligible()`
SQL-function path (used by the eval-gate self-test at `app/eval/gates
.py::check_repealed_as_current`) is a **synthetic isolated test**
(inserts a fake effect on an arbitrary component, checks it's excluded,
rolls back) — it does not scan real corpus data and will read `0` before
and after this task regardless. Rewiring `eligible_chunk_ids()` to
actually consult `lifecycle_effect` is a distinct future task (same
caveat repeated since AGENT-11/12/13/15 for commencement — now true for
repeal too).

## Assigned branch

`agent/repeal-extraction` (base: `dev`)

## Scope — four parts, all required

### A. `app/ingestion/repeal_extractor.py` (new)

Deterministic, no LLM — mirror `app/ingestion/enabling_extractor.py`'s
style and discipline (module docstring, always-emit-one-sentinel-per-
document, `ON CONFLICT DO NOTHING` idempotency).

- Detect the clean whole-act repeal clause. Start from the grounding
  regex above but verify/tune it yourself against the corpus — my
  157-document count is a starting point, not a spec. Confirm it does
  not fire on the savings sub-clauses or on the partial-दफा pattern
  before locking it in (re-run your count before/after, report the
  numbers — same discipline as AGENT-14).
- Resolve the named act to a `work` row. Reuse
  `enabling_extractor.py::_normalize_title`/`_resolve_work` — same
  problem (comma-normalize a Nepali act-name string, look up
  `work.title_ne`). Don't reimplement it.
- Explicit sentinel outcomes, matching `enabling_extractor.py`'s
  3-outcome shape: resolved (repealed work found in corpus), unresolved
  (regex matched, named act not in corpus), no match (no repeal clause
  found) — plus record the raw clause text (audit trail, same as
  `raw_clause_text` elsewhere).

### B. `app/authority/writer.py` — repeal proposal writer

- A new function (or an extension of `propose_lifecycle_commence` if it
  genuinely fits — your call, but its signature is single-component/
  single-effective-date and repeal is fan-out-across-components/
  cross-work-dependent, so a parallel function is more likely correct;
  state which you did and why) that, for a *resolved* repeal match:
  fans out a `pending`, `effect_type='repeal'` proposal to every
  `component.uri LIKE '{repealed_work.uri}/%'`.
- Dedup on `(component_uri, effect_type='repeal', approval_status='pending')`
  — same discipline as commence.
- No fabricated `legal_valid_time` lower bound — see the date-handling
  note above.
- Unresolved/no-match outcomes from part A do not call this — nothing to
  propose against an unknown component set.

### C. Wire into the pipeline + make proposals reviewable

- `app/ingestion/pipeline.py`: call the new extractor inside the
  existing `PROPOSE_LIFECYCLE` span (added by AGENT-12), alongside
  commencement extraction — same `SAVEPOINT`/`ROLLBACK TO SAVEPOINT`
  best-effort discipline, not a new stage.
- `scripts/review_lifecycle.py`: generalize the hardcoded
  `effect_type='commence'` filters so repeal proposals are listable and
  approvable through the same dual-sign-off CLI. Smallest change that
  achieves this — do not rewrite the CLI.

### D. Tests

- Corpus-count regression for the extractor (before/after any regex
  tuning, reported in your return message).
- `repeal_extractor.py`: resolved / unresolved / no-match sentinel
  cases, plus at least one case proving the savings sub-clauses
  ((२)/(३) etc.) and the partial-दफा pattern do **not** produce a
  whole-act repeal match.
- `writer.py`: fan-out-per-component, dedup, no-fabricated-date
  behavior.
- `review_lifecycle.py`: repeal proposals are listable/approvable
  through the CLI (extend existing test file's fixtures, don't
  duplicate its harness).

## Out of scope — do not touch

- Partial repeal of a specific दफा of another act (the harder 62-match
  sub-pattern above) — structurally different (targets a
  `component_uri` directly, not a whole work). Future task if ever
  prioritized.
- Text-substitution consequential amendments
  ("...शब्दहरूको सट्टा ... शब्दहरू राखिएका छन्").
- `<amend>` inline-tag correlation against a document's own
  "संशोधन गर्ने ऐन"/"संशोधन" amendment-history table — this is a
  separate, comparably-sized task (renumbered **AGENT-18**, see
  PROGRESS.md; the old plan called this "AGENT-16" before this task
  claimed that number for repeal specifically).
- Expiry/lapse/sunset-clause extraction — **grounded and dropped**: a
  broad `म्याद`/`स्वतः खारेज`/`कालावधि` keyword sweep returned 354 hits,
  all false positives on spot-check (generic "time limit"/"deadline"
  usage in procedural contexts, not act-level sunset clauses). No
  evidenced pattern in this corpus. Do not build extraction for a
  pattern that doesn't appear to exist — if you find a real one while
  grounding part A, stop and report it rather than silently expanding
  scope.
- `eligibility_gate.eligible_chunk_ids()` / retrieval rewiring to
  actually consult `lifecycle_effect` — separate future task, see
  honest scope note above.
- `work_relations` table — considered and rejected: its columns
  (`subordinate_work_id`/`enabling_work_id`) are enabling-power-specific
  and would be a confusing semantic fit for repeal; `lifecycle_effect`
  is the correct authority-bearing table and needs no schema change.

## Relevant System Design sections

- `system-design.md` §2 Core Invariant #1 (bitemporal store is single
  authority), #5 (human-gated dual approval — motivates part C's CLI
  fix), #6 (as-of/per-claim validity).
- PS-2 (no fabricated effective date — motivates the date-handling
  note), PS-3 (citations resolve to the authoritative instrument
  chain), PS-15 (status enum distinguishes repealed/spent/lapsed —
  this task populates the `repeal` half of that).

## Required checks

- `make test`
- `make lint`
- `make eval-gates` (zero-tolerance gates must stay at 0 — per the
  honest scope note, `repealed-as-current` is a synthetic self-test
  unaffected by this task either way; confirm it still reads 0 and
  report, but do not expect it to change).

## Explicitly forbidden changes

- No new migration / schema change (`lifecycle_effect.effect_type`
  already supports `'repeal'`; nothing here requires DDL).
- No changes to `eligibility_gate.py` or `postgres_retriever.py`.
- No changes to `<amend>`-tag handling, `parser.py`, or
  `commencement_extractor.py`.
- No changes to any file outside: `app/ingestion/repeal_extractor.py`
  (new), `app/authority/writer.py`, `app/ingestion/pipeline.py`,
  `scripts/review_lifecycle.py`, and their tests.

## Commit authorship

Every commit must be authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never AI-attributed, no
`Co-Authored-By: Claude` trailer, no "Generated with Claude" line. Use:

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
