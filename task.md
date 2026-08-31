# AGENT-17 — Fix दफा component-URI collisions (schedule + compound numbering)

**Branch:** `agent/schedule-header-collision`
**Base:** `dev`
**Engineer:** Pi

**Commit authorship — MANDATORY on every commit:**
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By` trailer. No AI attribution of any kind, ever.

---

## Objective

`app/authority/parser.py::parse_law()` still produces documents with **duplicate
`component.uri` values** even after AGENT-14's header-regex fix. This is a live
correctness gap in the bitemporal authority store (Core Invariant #1 — "the
bitemporal store is the single authority") and directly threatens PS-3
("Citations resolve to the authoritative instrument chain... CI reconstructs
every expression from base + effects... or it does not publish") — a
non-unique URI can't be cited precisely, and a future `lifecycle_effect`
proposal (commence/repeal) targeting that URI can't tell which of the
colliding provisions it means.

Fix the root causes in `parser.py` so distinct legal provisions never collide
onto one URI, add a VALIDATE-stage safety net that catches any future
unanticipated collision instead of silently persisting it, and clean up the
already-corrupted rows in the live DB.

## Grounding already done (re-verify counts, don't re-derive from scratch)

Ran `parse_law()` against all 677 records in `laws.jsonl` (repo root):

- **61/677 documents produce duplicate component URIs — 857 excess/collided
  rows total.** Reproduce with:
  ```python
  from app.authority.parser import parse_law
  import json
  from collections import Counter
  # for each record: uris = [c.uri for c in parse_law(rec).components]
  # flag any uri with Counter(uris)[uri] > 1
  ```
- Root-cause breakdown, corpus-verified by hand-inspecting the raw content
  around the colliding matches (three distinct patterns, not one):
  1. **53/61 docs — अनुसूची (schedule) bold-numbered list items misclassified
     as दफा headers.** Schedules appended at the end of an act reuse the exact
     same `**N. Title:**` bold format as दफा headers, restarting from 1 — e.g.
     `आयकर_ऐन_२०५८`: real `**१. संक्षिप्त नाम...**` near the top (offset ~1001)
     collides with a schedule's `**१. हासयोग्य सम्पत्तिको वर्गीकरण...**` under
     `अनुसूची-२**\n\n(दफा १९ सँग सम्बन्धित)` (offset ~252723). `_HEADER_RE` has
     no अनुसूची boundary marker at all today, so schedule list items are never
     recognized as schedule content — they fall straight into the bold-दफा
     alternative and get `component_type="dafa"` with numbers that collide
     with real दफा of the same number earlier in the document.
  2. **1/61 doc — compound "chapter.section" (N.M) दफा numbering collapses to
     N.** `आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५` numbers its दफा as `२.१`, `२.२`,
     ... `२.९` inside परिच्छेद-२. `_HEADER_RE`'s bold group
     `([०-९0-9]+)[\.।:]` stops at the first `.`, so all nine collapse onto
     `.../dafa/2` (this single document alone produces 9 same-URI rows).
     Corpus-wide: 26 compound-header instances across 5 documents total (not
     all 5 collide — some compound docs don't have a colliding plain-number
     दफा already at that number).
  3. **7/61 docs — genuine same-number, different-content दफा collisions in
     the main body**, unrelated to either pattern above. E.g.
     `कारागार_ऐन_२०७९` has two separate bold `**४१. ...**` headers with
     completely unrelated titles/content ("खुला कारागार सम्बन्धी व्यवस्था" vs
     "मुद्दा हेर्ने अधिकारी"); `कम्पनी_ऐन_२०६३` has two `**१८४. ...**` headers
     ~3800 chars apart, also substantively different. No परिच्छेद-restart or
     अनुसूची boundary explains these — looks like a numbering error in the
     underlying source text itself. **Do not try to guess which one is
     "correctly" numbered** — that's not decidable from the text and isn't
     this task's call (Prime Directive: a loud refusal beats a quiet wrong
     answer). The fix must preserve both provisions distinctly.
     Other docs in this bucket: `कसूरजन्य_सम्पत्ति_तथा_साधन_(रोक्का_नियन्त्रण_र_जफत)_ऐन_२०७०`,
     `उपभोक्ता_संरक्षण_ऐन_२०७५`, `नेपाल_स्काउट_ऐन_२०५०`,
     `प्राविधिक_शिक्षा_तथा_व्यावसायिक_तालीम_परिषद्_ऐन_२०४५`,
     `मुलुकी_देवानी_संहिता_२०७४`, `मुलुकी_अपराध_संहिता_२०७४`.

- `upsert_expression()` (`app/authority/writer.py`) dedups only on
  `(component_uri, as_of, text_hash)` — it does **not** enforce one row per
  `(component_uri, as_of)`. So today this doesn't silently drop text (each
  colliding provision gets its own `expression` row), but it does leave the
  URI itself ambiguous — nothing downstream can tell which expression row is
  "the" दफा N without inspecting all of them.
- Two documents also have mismatched `<amend>`/`</amend>` tag counts
  (`नेपाल_कानून_व्यवसायी_परिषद्_ऐन_२०५०` has a corrupted `</amन>` close tag;
  `पेट्रोलियम_नियमावली_२०४१` has one genuinely unclosed `<amend>`). **Out of
  scope for this task** — only 2/677 docs, thin evidence, different failure
  mode (stray annotation text, not URI collision). Noted for a future pass,
  not carried into this brief.
- `document_type` in the corpus is only ever `act` or `regulation` (359 /
  318 of 677) — no evidence of other doc types needing VALIDATE support.
  **Not carried into this task.**

## Required fix (three parts)

1. **`app/authority/parser.py`** — stop misclassifying schedule content as
   दफा components. Detect an अनुसूची boundary (schedules consistently appear
   after all substantive दफा content in every example seen) and don't apply
   the bold-दफा alternative past it; give schedule content its own
   `component_type` (e.g. `"anushuchi"`) instead of `"dafa"`. Handle the
   compound N.M case so the full number (e.g. `"2.1"`) becomes part of the
   URI instead of collapsing to `"2"`. For the remaining genuine same-number
   collisions (pattern 3) that no reclassification can resolve: make sure the
   parser never emits two components with the identical URI — pick and
   document a deterministic disambiguation (e.g. an occurrence suffix on the
   URI) so both provisions stay distinctly addressable and neither silently
   overwrites the other downstream.
2. **VALIDATE stage** (`app/ingestion/pipeline.py`, laws path, ~line 234) —
   after parsing, assert the resulting component URIs are unique. This is a
   safety net for any numbering pattern not in this corpus today, not a
   substitute for part 1. Decide reject-vs-flag and justify the choice in
   your writeup; either way it must not silently pass a colliding document
   through unnoticed the way today's `has_dafa` boolean check does.
3. **Cleanup script for the live DB** — re-derive expected state from the
   fixed parser per document, the same way
   `scripts/cleanup_stale_authority_expressions.py` (AGENT-14) does. Reuse
   that script's pattern and its lifecycle-status guard (an orphan/incorrect
   URI with any non-`pending` `lifecycle_effect` row must be blocked and
   reported, never auto-deleted — Core Invariant #5 territory, a human
   decision is needed there, not a script). Extending that existing script is
   preferred over writing a new one from scratch if the shapes fit —
   Ponytail gate: reuse before new file.

**Optional, trivial, only if touching this area doesn't cost extra review
weight:** `app/ingestion/tariff_chunker.py:38` has a bare `5000` magic
threshold in `is_tariff_dominant()`. If you have a spare two lines, name it
(e.g. `_TARIFF_DOMINANT_HS_CODE_THRESHOLD = 5000`) with a one-line comment.
Skip it entirely if it would blur the diff — this task's real weight is the
URI-collision fix, don't let a drive-by change dilute review of that.

## Scope / allowed files
- `app/authority/parser.py`
- `app/ingestion/pipeline.py` (VALIDATE stage only — laws path)
- `scripts/cleanup_stale_authority_expressions.py` (extend) or a new script
  in `scripts/` only if the existing one's shape genuinely doesn't fit —
  justify in your writeup if you add a new file instead of extending
- `tests/test_parser.py`, `tests/test_ingestion_pipeline.py`,
  `tests/test_cleanup_stale_authority_expressions.py` (or equivalent)
- `app/ingestion/tariff_chunker.py` — optional 2-line constant extraction only

**Forbidden:** schema/migration changes (no evidence one is needed —
`component_type` is already free-text, `"anushuchi"` is just a new string
value); touching `eligibility_gate.py`, `validation_gate.py`, or any
retrieval-path file (out of scope, unrelated); touching
`repeal_extractor.py` / `commencement_extractor.py` / `enabling_extractor.py`
(different subsystem, already correct for this concern); the `<amend>` tag
mismatch issue noted above (explicitly out of scope, see grounding section).

## Required checks
- `make test`
- `make lint`
- `make eval-gates` — zero-tolerance gates (`repealed-as-current`,
  `not-yet-effective-as-current`, `overruled-as-good-law`) must stay at 0
- Corpus-wide before/after count: duplicate-URI document count and excess-row
  count (the 61 / 857 baseline above) — confirm it drops to 0 duplicate URIs
  across all 677 `laws.jsonl` records after the parser fix, using the same
  method as the grounding section
- Live-DB cleanup script run with `--dry-run` first, then for real; report
  before/after counts the same way AGENT-14's entry in `.agent/PROGRESS.md`
  did (component/expression/lifecycle_effect row counts, orphan counts, any
  lifecycle-status-guard blocks encountered)

## Zero-tolerance gates guarded
`repealed-as-current = 0`, `not-yet-effective-as-current = 0` — this task
touches `parser.py`/VALIDATE, which upstream feeds `lifecycle_effect` via
`propose_lifecycle_commence`/`propose_lifecycle_repeal`. Confirm neither gate
regresses.

## Self-review checklist before pushing
- No path skips the eligibility gate (this task doesn't touch it — confirm
  you didn't accidentally add a new retrieval branch)
- Model never writes citations (unaffected — this is ingestion-side only)
- No new schema/table/column, no new abstraction beyond what's justified
  above (Ponytail)
- Every colliding-URI case found in the corpus grounding above is now either
  correctly reclassified (schedule/compound) or deterministically
  disambiguated (genuine source collisions) — not just "fewer" collisions,
  zero, verified by the full 677-record count

## Return
Commit hash(es), changed files, checks run/results (paste the before/after
677-record duplicate-URI count), live-DB cleanup counts, assumptions made
(especially your अनुसूची boundary-detection regex and disambiguation scheme
for pattern 3), and any remaining risks.
