# Wakil-G — Orchestration Progress

## Current task
AGENT-22 — wire citation rendering + temporal revalidation to real
authority. Branch: `agent/authority-linked-citations` (re-cut fresh off
`dev`, same scope as the original brief). Assigned to Pi.

## Status
**ASSIGNED (AGENT-22, re-dispatched)** — task.md re-pushed with an added
note asking the engineer to verify `git status` on this exact branch
before reporting completion, given what happened last time.

- **AGENT-22/23 dispatch mix-up (2026-09-01)**: Pi reported AGENT-22
  complete (migration, `validation_gate.py` rewiring, temporal-authority
  check, backfill — specific, detailed, plausible-sounding) with all
  checks green. On checkout, **none of it existed anywhere in the repo**
  — no migration file, no diff to `validation_gate.py`, `pgvector_indexer.py`,
  or `backfill_authority_layer.py` on that branch or any other. What
  *was* sitting uncommitted in the shared working tree, on the wrong
  branch (`agent/authority-linked-citations` instead of
  `agent/small-safety-fixes`), was a complete and correct implementation
  of **AGENT-23** — confirmed by diffing it against AGENT-23's own
  task.md item-by-item, all three items present and correct. Read this as:
  AGENT-23 was actually done, well, and then reported under AGENT-22's
  name while checked out on AGENT-22's branch — not a partial fix, not a
  smaller version of AGENT-22, a completely different task's work
  described as if it were the assigned one. This is a step beyond the
  uncommitted-diff pattern seen on AGENT-15/19/20 (real work, just not
  committed) — here the specific thing reported does not exist at all.
  Recovered the misplaced work with `git stash` → checkout the correct
  branch → `git stash pop`, then ran the full Claude Review Gate on it.
  AGENT-22 itself was never implemented and needs to be dispatched again
  from scratch. Flagging this plainly to Prakash rather than quietly
  re-running it, given how it happened.

- **AGENT-23 (2026-09-01, MERGED)**: once recovered onto its correct
  branch, all three items present and correct. (A) `review_documents.py`:
  `redaction_failed` added to the `FOR UPDATE` select, refuses approval
  before touching approver state — new test confirms the guard fires
  first (`a1 is None`, status still `pending`, rollback recorded). (B)
  `scripts/backfill_source_kind.py` (new): single idempotent `UPDATE`,
  dry-run/real-run/idempotency all tested. (C) `pipeline.py::_content_hash()`
  now digit-folds (reused the module's existing `_DEVANAGARI_DIGITS`
  table rather than importing `pii_redactor.py`'s private helper — better
  than what the brief suggested) and canonicalizes whitespace
  (`re.sub(r"\s+", " ", ...).strip()`), plus the required companion
  `scripts/recompute_content_hashes.py` (new) — recomputes
  `documents.content_hash` in place without touching `ingestion_status`/
  approvers, confirmed by a dedicated test
  (`test_recompute_updates_hash_only_not_status`). Two existing tests that
  used to duplicate the old hash formula inline were updated to call
  `_content_hash()` directly instead — won't silently drift from the real
  implementation again.
  Independently re-verified: `make test` (176 passed, 3 skipped —
  matches), `make lint` clean, manually ruff/mypy'd the two new scripts
  (not in the Makefile's fixed list, same pre-existing gap as
  `review_lifecycle.py`/`review_documents.py`) — clean, `make eval-gates`
  all three zero-tolerance gates at 0.
  Merged `agent/small-safety-fixes` → `dev` (`--no-ff`). Deleted the empty
  `agent/authority-linked-citations` branch (only ever had the task.md
  brief commit — no real work was ever committed to it, confirmed before
  deleting).

- **Follow-up review (2026-09-01)**: Prakash brought 6 more findings after
  AGENT-19/20/21 merged. Verified all 6 against current code before
  answering, not from memory (several touch things AGENT-20 deliberately
  left alone): (1) `validation_gate.py` still resolves every claim purely
  against `chunks`, never `component`/`source_publication`/
  `lifecycle_effect` — confirmed the biggest remaining gap, root-caused to
  chunks having no link back to the authority tables at all; (2)
  `upsert_source()`'s early-return-on-existing-row means `derived_verified`
  (AGENT-20) only applies to new rows — confirmed; (3)
  `review_documents.py` doesn't check `redaction_failed` before approving
  — confirmed, not live-exploitable today (NKP-only column, NKP locked
  out) but a real hole in the tool; (4) `PERSIST_AUTHORITY` still runs
  before document approval — confirmed, unchanged, and its risk is coupled
  to (1): once citation rendering starts reading `component`/`expression`,
  it must also re-check document-approval status or (4) becomes newly
  exploitable; (5) `_citation()`'s `source_kind`/`ocr_confidence` are
  wrong-column reads, not `source_publication.kind`/`documents
  .ocr_confidence` — same root cause as (1); (6) `_content_hash()` only
  NFC-normalizes, missing the digit-fold + whitespace canonicalization
  `docs/ingestion_design.md` specifies — confirmed, found a reusable
  `_digit_fold` pattern already in `pii_redactor.py`, and flagged that
  fixing this changes the hash for the whole already-ingested corpus
  (needs a recompute backfill, not a bare function change, or every
  document reads as "amended" on the next ingest run).
  Design decision made without a further round-trip, per Prakash's
  explicit request to stop drip-feeding this: closing (1)/(5) needs two
  small schema additions — `chunks.component_uri` and
  `documents.source_pub_id` — both just persisting values the pipeline
  already computes locally and discards today, not new extraction logic.
  Scoped as AGENT-22 (the schema + citation-rendering + new live
  repeal/expiry/suspend check — the actual "revalidate temporal validity
  against authority" enforcement Core Invariant #6 requires and nothing
  in the codebase does today) and AGENT-23 (the three small independent
  fixes — (2)/(3)/(6) — bundled together only because each is too small
  for its own branch, not because they share a theme). (4)'s coupling to
  (1) is called out explicitly in AGENT-22's brief so the engineer keeps
  the existing `eligible_chunk_ids()`-gated flow rather than
  reintroducing a pre-approval leak while rewiring citation rendering.

- **AGENT-21 (2026-09-01)**: relabeled `ingest_laws.py`'s `"processed"`
  count/print to `"pending_review"`. This diff was written by Pi while
  working on AGENT-20 (see below) — relocated to its correct branch by
  Claude rather than sent back for a trivial trim, since the content was
  already correct and already independently checked. Re-ran `make lint`/
  `make test` on this branch alone after applying it (162 passed, matches
  the branch's pre-AGENT-20 base) before merging.
  Merged `agent/ingest-status-messaging` → `dev` (`--no-ff`).

- **AGENT-20 (2026-09-01)**: Pi returned approved-only eligibility +
  `scripts/review_documents.py` + the `derived_verified` relabel, again
  sitting uncommitted in the working tree (third time this exact failure
  mode has happened across AGENT-15/19/20 — worth a standing fix to how
  engineers are told to finish a task, not just noting it each time).
  **Scope violation found and corrected before merge**: the diff also
  modified `scripts/ingest_laws.py` — AGENT-21's exclusive scope on a
  separate branch, not in AGENT-20's allowed-file list. Content was
  correct (the identical `"processed"`→`"pending_review"` rename AGENT-21's
  brief asked for) but the wrong branch. Extracted that hunk out
  (`git diff` → `git checkout --`), verified the remaining AGENT-20 diff
  still passed all checks on its own, then applied the extracted hunk to
  `agent/ingest-status-messaging` directly (see AGENT-21 entry above) —
  mechanical relocation of an already-reviewed patch, not new engineer
  work, so no second Pi round-trip.
  Independently re-verified the corrected diff rather than trusting the
  report: `make test` (169 passed, 3 skipped — matches), `make lint`
  clean, `make eval-gates` all three zero-tolerance gates at 0. Manually
  ruff/mypy'd `scripts/review_documents.py` + its test (not in the
  Makefile's fixed lint list, same pre-existing gap as `review_lifecycle.py`)
  — clean. Read `review_documents.py` end-to-end against
  `review_lifecycle.py`'s proven dual-approval pattern (`FOR UPDATE` row
  lock, distinct-approver enforcement, satisfies the
  `documents_dual_approval` CHECK constraint by construction before the DB
  ever has to reject anything) — faithfully mirrored, correctly adapted for
  `documents`' TEXT approver columns vs. `lifecycle_effect`'s UUID ones.
  `tests/test_eligibility_gate.py` gained a real behavioral harness
  (`FilteringCursor`/`FilteringConn` that actually evaluates the predicate
  against seeded rows) replacing the old string-only SQL-text assertions —
  a genuine improvement over what the brief asked for, not just satisfying
  it; the two tests that used to hard-assert `'pending'` inclusion were
  correctly rewritten to assert exclusion, not left contradictory.
  **Correction to a prior diagnostic-pass answer**: `no_commencement_clause`
  turned out to already be a distinct, visible sentinel (see the note
  above dated before this task was scoped) — dropped from this task before
  it started, not discovered mid-review.
  Merged `agent/document-approval-gate` → `dev` (`--no-ff`).

- **AGENT-19 (2026-09-01)**: Pi returned a one-line fix — `eligibility_gate.py::eligible_chunk_ids()`

- **Correction (2026-09-01, before AGENT-20 was scoped)**: the diagnostic
  pass's answer to "should `no_commencement_clause` be a distinct, visible
  outcome" was wrong — re-reading `commencement_extractor.py` while
  scoping AGENT-20 found it already is. The module docstring says so
  outright (*"Unknown/no-match cases emit a sentinel proposal instead of
  disappearing silently"*), `classify_commencement()` returns
  `CommencementProposal(None, "no_commencement_clause", "")` as its
  explicit fallback, and AGENT-13's live-DB backfill run confirms real
  rows exist with this value (95 of them). Dropped from AGENT-20's scope
  since there's nothing left to fix. Also dropped a second item that would
  have been dead work without a schema change: making
  `validation_gate.py::_citation()` read the real `source_publication.kind`
  instead of `chunks.source_type` — there's no FK today linking a
  document/chunk to the specific `source_publication` row backing it, so
  this needs a schema decision, not a query fix. Flagged to Prakash below,
  not built.

- **AGENT-19 (2026-09-01)**: Pi returned a one-line fix — `eligibility_gate.py::eligible_chunk_ids()`
  gained `AND c.source_type <> 'nkp_case'` — plus a matching test. The diff
  was sitting uncommitted in the working tree (same failure mode noted in
  AGENT-15); verified the full diff content and all three checks
  independently before committing it myself (author `Prakash Basnet`, per
  policy): `make test` (162 passed, 3 skipped — matches), `make lint`
  (ruff + mypy --strict clean), `make eval-gates` (all three zero-tolerance
  gates at 0). Went further than the reported checks: traced every direct
  `FROM chunks`/`JOIN chunks` query in the codebase (not just the ones the
  brief named) to confirm the fix is actually complete, not just
  plausible — `postgres_retriever.py`'s vector/lexical queries and
  `gated_orchestrator.py::_resolve_cross_refs` both filter by the
  `eligible_chunk_ids()` result set; `validation_gate.py::validate_and_render`
  independently **recomputes** `eligible_chunk_ids()` itself rather than
  trusting retrieval's set (real defense-in-depth, better than the brief
  assumed); `query_graph.py::_fetch_enabling_chunk` can't structurally reach
  an `nkp_case` chunk regardless (`nkp_case` rows always have `work_id IS
  NULL`, and that path joins on a specific Act's `work_id`) — safe by
  construction, not by an explicit filter. **Found but out of scope, not
  blocking**: `_fetch_enabling_chunk`'s final chunk fetch has no
  eligibility filter at all (no `ingestion_status`/`effective_date` check),
  a pre-existing gap unrelated to `nkp_case` — co-retrieved enabling
  provisions can bypass the gate entirely. Not touched here (AGENT-19 was
  scoped to the `nkp_case` exclusion only); flagged for a future task, only
  becomes one if Prakash asks.
  Merged `agent/nkp-precedent-lockout` → `dev` (`--no-ff`).

- **Diagnostic pass (2026-09-01)**: Prakash asked 15 grounding questions
  before rating/fixing an external review of the ingestion/gate design.
  Answered each against `system-design.md` + the live code (not guessing),
  citing file:line for every code claim. Found three **live** violations of
  already-approved invariants, not "not built yet" gaps: (1) Core Invariant
  #5 / PS-2 — `eligibility_gate.py::eligible_chunk_ids()` returns
  `ingestion_status IN ('approved', 'pending')`, so unreviewed documents are
  retrievable today (a test, `test_eligible_chunk_ids_includes_pending_valid_document`,
  explicitly locks this in — not accidental drift); (2) §6/PS-1 — `nkp_case`
  chunks are tiered and retrievable via the same gate with zero
  `overruled-as-good-law` check anywhere in the query path, contradicting
  §6's explicit "ingested but not answered from" rule for pre-Phase-D case
  law; (3) no document-level dual-approval path exists anywhere in the
  codebase (`review_lifecycle.py` only approves `lifecycle_effect` rows) —
  `documents.ingestion_status` can never legitimately reach `'approved'`
  today. Also found a PS-3 mislabel (`writer.py::upsert_source` hardcodes
  `kind='official_copy_unverified'` for `laws.jsonl`, which
  `docs/ingestion_design.md` §1.2 itself calls a third-party consolidation,
  not an official/unverified-original copy) and confirmed the source PDF
  URLs in `laws.jsonl` are never fetched by any code path — `content` is the
  entire corpus this pipeline ingests from.
  Prakash confirmed a fix plan across three tasks, reordered from his
  original proposal once file-overlap was checked (both the pending-gate
  fix and the NULL-effective-date fix land in the same function,
  `eligible_chunk_ids()` — sequencing beats true parallelism here to avoid
  two engineers colliding on one query):
  - **AGENT-19 (MERGED to dev, 2026-09-01)** — exclude `nkp_case` at the
    gate. See entry above for full detail.
  - **AGENT-20** (this task, branched now that AGENT-19 is merged, so it
    lands cleanly on the same function without conflicting) — `eligible_chunk_ids()` to
    `ingestion_status = 'approved'` only, NULL `effective_date_ad` excluded
    unless an approved `commence` lifecycle_effect resolves it (join
    directly, don't trust the denormalized cache column — §7.4 already
    warns against that), plus a new `scripts/review_documents.py`
    dual-approval CLI (mirrors `review_lifecycle.py`'s shape, doesn't
    extend it — different table, different approver-identity shape, would
    make the CLI's "approve X" ambiguous). Also folds in: `commencement_extractor`
    surfacing `no_commencement_clause` as a distinct, visible outcome
    (currently indistinguishable from "nothing to extract"), and
    `source_publication.kind` → `derived_verified` for `laws.jsonl` sources
    at approval time (not `verified_internal_consolidation` — this system
    didn't produce the consolidation, a third party did).
  - **AGENT-21** (queued, no file overlap with 19/20, can run anytime) —
    `scripts/ingest_laws.py` CLI reporting currently calls a `pending`
    document "processed"/"ingested" — cosmetic messaging fix, no gate risk,
    lowest priority.
  Two other findings from the diagnostic pass were surfaced but **not**
  turned into tasks (don't clear the bar per [[feedback_task_creation_bar]]
  on their own): `PERSIST_AUTHORITY` writes `work`/`component`/`expression`
  before document approval, contrary to §5's stated ordering — acceptable
  for now since nothing reads component/expression as proof of
  searchability (only `documents`/`chunks`, confirmed by grep), *provided*
  AGENT-20's approval-CLI task adds a regression test pinning that a
  component on an unapproved document is never citable through
  `validation_gate.py`; and lifecycle-proposal-extraction failure staying
  non-blocking (confirmed still correct given AGENT-20's NULL-exclusion
  fix makes a failed extraction equivalent in effect to "nothing to
  extract" — no correctness gain from quarantining the whole document).

- **AGENT-18 review round 2 (2026-08-31)**: Pi returned `d99712a` fixing
  both round-1 findings. Independently re-verified rather than trusting
  the report: re-ran the extractor against all 677 records (exact match
  on every reported number — 451 docs, 7,358 proposals [1,941 named-act /
  5,417 ordinal], 8,390 skipped, `unresolved_ordinal`/`unresolved_unknown-
  ordinal` both absent/0), `make test` (161 passed, 3 skipped), `make
  lint`, manual ruff/mypy on `amend_extractor.py` (zero issues, no
  pre-existing baseline to compare against since the file is new this
  task), `make eval-gates` (all three zero-tolerance gates at 0). Hand-
  traced the line-wrap fix line-by-line against the exact real document
  that surfaced the bug (महान्यायाधिवक्ताको_पारिश्रमिक_...ऐन_२०५२) —
  confirmed all 5 real rows now parse correctly, including the boundary
  logic that stops a continuation line from swallowing the *next* row
  (a `<=2`-digit guard on row-start markers plus an immediate
  require-date finalize check after every line). Checked the ordinal
  normalization for the failure mode a spelling-unification pass risks —
  two different ordinal numbers silently colliding onto the same
  normalized key — by running `_normalize_ordinal` over all 41
  `ORDINAL_DAYS` entries and confirming zero collisions (41 distinct
  words → 41 distinct normalized keys, each with its original numeric
  value intact). The corpus-count deltas all reconcile: `unresolved_named-
  act` 667→570 and `no_table` 477→187 (the latter's large drop makes
  sense once understood — a table whose *every* row happened to wrap
  previously came back fully empty, misclassified as "no table" rather
  than "table exists, rows recovered"), `duplicate`/`unresolved_other`
  both rose slightly as a direct, explainable consequence of more tags
  now reaching table lookup instead of being swept into `no_table` before
  classification ever ran. New tests use the real document's content
  (not a synthetic string) for the wrap case and 4 of the 55 real variant
  tokens for the ordinal case. Merged `agent/amend-tag-correlation` →
  `dev` (`--no-ff`).

- **AGENT-18 review round 1 (2026-08-31)**: Pi returned `e73103a` — 438
  docs with proposals, 6,424 proposals (1,854 named-act / 4,570 ordinal),
  9,324 skipped. Independently re-verified rather than trusting the
  report: re-ran the extractor against all 677 records via a stub `conn`
  (exact match on every reported number), `make test` (159 passed, 3
  skipped), `make lint`, manual ruff/mypy on the 3 ingestion-path files
  (4 pre-existing pipeline.py errors, unchanged from base), `make
  eval-gates` (all three zero-tolerance gates at 0). Hand-verified the
  dedup design against a real document (सुशासन_(व्यवस्थापन_तथा_सञ्चालन)_
  ऐन_२०६४'s दफा ३ — two tags at nearby offsets really are the same
  amending act touching two different उपदफा within one दफा, correctly
  collapsed to one दफा-level fact, not a bug) and the `unresolved_named-
  act` abstention against another (महाभियोग_(कार्यविधि_नियमित_गर्ने)_ऐन_
  २०५९ — the tag genuinely names an act absent from this document's own
  table, correctly abstained, not fabricated).
  But found two real, in-scope, fixable causes hiding inside two of the
  skip buckets rather than genuine abstentions: (1) `_ROW_RE`'s name
  group excludes newlines, silently dropping any amendment-table row
  whose act name wraps onto a second line — confirmed directly against
  `महान्यायाधिवक्ताको_पारिश्रमिक_सेवाको_शर्त_र_सुविधा_सम्बन्धी_ऐन_२०५२`
  (5 real table rows, only 2 captured), quantified corpus-wide at 68/498
  documents (13.7%) with at least one dropped row, 135/1,952 rows (6.9%)
  missing — feeds both `unresolved_named-act` (667) and
  `unresolved_ordinal` (225), indistinguishable today from a genuine
  "not in this table" abstention; (2) 936 of ~5,731 ordinal-shaped tags
  (16%) use a chandrabindु/anusvara or missing-trailing-nasal spelling
  variant of an already-known `ORDINAL_DAYS` word (`पाँचौ` vs `पाँचौं`
  alone is 254 instances) — a normalization gap, not a new-word problem;
  55 distinct unknown tokens total. Both are quantified, bounded,
  in-scope fixes (not new adjacent problems — this is the task's own
  named mechanism under-delivering on its own grounding, same character
  as AGENT-17 round 1, not a candidate for demotion under
  [[feedback_task_creation_bar]] since it's about this task's own
  deliverable, not a new one). Rework note appended to `task.md`
  (`a07f22d`) with exact regex diagnosis and re-verification asks; same
  branch, same engineer.

- **AGENT-18 scoping (2026-08-31)**: assigned per Prakash's explicit
  request (clears the task-creation bar — see [[feedback_task_creation_bar]]
  memory going forward). Re-verified the old backlog note's grounding
  before writing the brief rather than trusting it as-is — it materially
  undersold the real shape: the note said "83% dominant named-act
  pattern, ordinal position secondary." A precise per-tag classification
  of all 15,748 `<amend>` instances (677 records) found the *opposite*
  emphasis — named-act 31.3% (4,926), ordinal-position 55.9% (8,808, the
  actual majority), gazette-date-only 0.2% (24, thin, abstain), and a
  12.6% (1,990) "other" bucket that turned out to be mostly *not*
  amendment records at all (commencement dates already covered by
  AGENT-12, repeal asides already covered by AGENT-16, name changes,
  unrelated constitutional cross-refs, pay-adjustment decisions) —
  explicitly scoped out, don't guess. Also found and grounded a
  previously-undocumented parsing surface both patterns depend on: an
  ordered amendment table near the top of each document (`संशोधन गर्ने
  ऐन` for acts, bare `संशोधन`/`संशोधन गर्ने नियम` for regulations —
  different heading, same shape), covering 496/519 (96%) of amend-tag
  documents once both heading variants are recognized; 23/519 (4%) have
  no table at all — abstain, don't fabricate. Pointed the brief at two
  direct reuse opportunities to keep this extend-existing per Ponytail:
  `commencement_extractor.py::ORDINAL_DAYS` for the ordinal-word table
  (same words, reused as position instead of day-offset) and
  `enabling_extractor.py::_normalize_title()` for act-name matching
  against the table (not `_resolve_work()` — matching against the
  document's own table, not the `work` table). `effect_type='amend'` and
  `EffectType.AMEND` already exist (migration 001, `models.py`) — no
  schema change. Explicit honest-scope note baked into the brief: this
  only records amendment *facts* (which दफा, by which act, when) — this
  corpus has no pre-amendment text to version, so it doesn't achieve full
  PS-17 compliance (closing a prior expression's valid_time), same
  "don't fabricate what the corpus doesn't support" discipline as every
  prior lifecycle-extraction task.

- **AGENT-17 review round 2 (2026-08-31)**: Pi returned `461fcb9` addressing
  both round-1 findings. Independently re-verified rather than trusting the
  report: re-ran the 677-doc corpus check (0 duplicate URIs, confirmed),
  `make test` (153 passed, 3 skipped, matches), `make lint` clean, manual
  ruff/mypy on the 3 ingestion-path files (same 5 pre-existing errors,
  confirmed byte-for-byte unchanged), `make eval-gates` (all three
  zero-tolerance gates at 0). Then specifically re-checked the two flagged
  documents rather than just the aggregate count: `आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५`
  now parses cleanly to `2.1`...`2.9` with zero `/occurrence/` suffixes —
  gap 1 fully fixed, confirmed against the real record (the new test loads
  it directly from `laws.jsonl` instead of a synthetic string, exactly as
  asked); `स्टाण्डर्ड नाप र तौल नियमहरु २०२७` now recognizes 185 `anushuchi`-typed
  components (was ~0) — gap 2 substantively fixed, अनुसूची-boundary docs
  240→250 corpus-wide (modest increase is legitimate: most of the
  remaining 389-word-mentions were plain cross-references in body text,
  not real schedule headers, same discipline as AGENT-14's cross-reference
  exclusion).
  **New pattern surfaced by this deeper check, not part of round 1's
  findings**: 590 components across 78 documents (down from 601/80) are
  still `/occurrence/`-suffixed — but now concentrated in a different,
  deeper structural pattern than either of round 1's two gaps: 7 documents
  (down from 8 — आयुर्वेद's compound fix resolved cleanly) are the original
  genuine no-schedule same-number collisions (by design, matches the task
  brief's "disambiguate, don't guess" scope); the remainder is dominated by
  5 technical/tabular regulation schedules (`स्टाण्डर्ड नाप र तौल नियमहरु
  २०२७` 115, `भन्सार_महसुल_ऐन_२०८१` 69, engineering/health/education/
  insurance नियमावली 14-26 each) whose schedules contain their **own
  internal recursive numbering restarts** — e.g. एक अनुसूची with several
  sub-tables each independently renumbering from 1, producing URIs like
  `/anushuchi/10.5.9` (schedule 10 → item 5 → its own compound sub-item 9)
  that still collide across sub-tables within the same schedule. The
  disambiguation net absorbs this correctly (0 duplicate URIs holds either
  way, Core Invariant #1 intact, no citation can ever resolve
  ambiguously) — this is a classification-precision gap for a narrow set
  of tabular technical schedules, not a data-integrity risk, and it's a
  structurally different, deeper problem than what this task's grounding
  or either rework round scoped (nested/recursive schedule numbering, not
  top-level दफा-vs-अनुसूची misclassification or one-level compound
  numbering). **Accepted as-is, not sent for a third round, and not given
  a task number** — the two specific gaps this task actually found and
  scoped are both genuinely fixed on their real evidenced examples; the
  remainder is a citation-precision nice-to-have with no Core Invariant or
  gate risk, doesn't clear the bar for a new AGENT-N task on its own
  (logged as an informational finding below, not a planned task — only
  becomes one if Prakash asks).
  Merged `agent/schedule-header-collision` → `dev` (`--no-ff`, matches
  AGENT-N merge-commit convention).

- **AGENT-17 review round 1 (2026-08-31)**: Pi returned `cf3c1aa` claiming
  0/677 duplicate-URI docs, `make test`/`make lint`/`make eval-gates` all
  green. Independently re-verified rather than trusting the report (Claude
  Review Gate) — re-ran the 677-doc corpus check myself (confirmed 0
  duplicates), re-ran `make test`/`make lint`, and manually ruff/mypy'd the
  3 ingestion-path files not in the Makefile's fixed list (5 pre-existing
  mypy errors, byte-for-byte unchanged from base, confirmed by diffing
  before/after). But went further than trusting the aggregate 0-duplicate
  count: checked *why* it was 0, since a disambiguation safety net can mask
  a broken primary fix, not just verify one exists. Found the compound-N.M
  fix and the अनुसूची-boundary fix — the two headline root-cause fixes named
  in the task brief — don't actually fire on real corpus data, only the
  disambiguation net (`/occurrence/N` suffixing) is doing the work: (1) the
  task's own grounding example, आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५, still
  collapses २.१-२.९ onto plain "2" because its real header format has no
  punctuation immediately after the compound number (`**२.१ परिषद्‌को
  स्थापना :**`) — the added regression test used a synthetic string with a
  period right after the number, which isn't how this document (already
  quoted verbatim in task.md's own grounding section) is actually
  formatted, so the test passed without exercising the real case; (2) the
  अनुसूची-boundary regex requires the dash to immediately follow अनुसूची with
  no whitespace tolerance, missing most real corpus formatting variants
  (`अनुसूची - ३`, `अनुसूची ३ (ख)`, `अनुसूची १२` — all miss) — only 240/677 docs
  get a boundary recognized despite 389 containing the word, and 72 of the
  80 still-`/occurrence/`-suffixed documents do contain schedules the
  regex should have caught (601 occurrence-suffixed components corpus-wide
  vs. the 7-8 genuine no-schedule collisions this net was meant to cover).
  Data integrity holds either way (0 duplicate URIs, no citation could ever
  resolve ambiguously) — this is a correctness-of-classification gap, not
  a data-loss risk, but two of three named fixes don't do what the task
  and Pi's own summary claim. Rework note appended to `task.md`
  (`d10496e`) with exact regex diagnosis and a re-verification ask; not
  re-scoped, not re-assigned — same branch, same engineer, per the Rework
  Loop.

- **AGENT-17 scoping (2026-08-31)**: session-start hygiene pass first — 13
  fully-merged `agent/*` working branches deleted (all confirmed ancestors of
  `dev`), stray `task.md.bak` deleted; `.agent/extract_meta_bottleneck.md` and
  `docs/legal_rag_ingestion_best_practices.md` (both flagged untracked since
  AGENT-15, origin unconfirmed) kept per Prakash's call, still untracked.
  Then picked AGENT-17 (VALIDATE-stage hardening) off the backlog per
  Prakash's direction, but corpus grounding found the backlog's framing
  ("duplicate section numbering" as a VALIDATE-stage gap) undersold the real
  bug: re-ran `parse_law()` against all 677 `laws.jsonl` records and found
  **61 documents still produce duplicate `component.uri` values post-AGENT-14
  — 857 excess/collided rows**, root cause is in `parser.py` itself, not
  just a missing VALIDATE check. Diagnosed three distinct patterns by hand
  (not just regex counts — traced raw content around each collision):
  53/61 = अनुसूची (schedule) bold-numbered list items with no dedicated header
  marker falling into the bold-दफा alternative and colliding with real दफा of
  the same number (e.g. आयकर_ऐन_२०५८'s अनुसूची-२ item "१." collides with the
  real दफा १ near the top); 1/61 = compound "N.M" चapter.section दफा
  numbering (आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५'s २.१–२.९) collapsing to just
  "N" at the first `.`; 7/61 = genuine same-number-different-content दफा
  collisions in the main body with no explaining pattern (कारागार_ऐन_२०७९'s
  दफा ४१ appears twice with unrelated titles, likely a source-text numbering
  error) — explicitly scoped as "disambiguate, don't guess which is correct"
  per the Prime Directive. Also found and explicitly dropped two thin-evidence
  items during the same pass: 2/677 docs with mismatched `<amend>` tag counts
  (real but too rare to justify a check — noted for a future pass, not
  carried forward) and doc-type support (corpus only has `act`/`regulation`,
  zero evidence of a gap — dropped, matches the established pattern of not
  carrying unevidenced items forward, e.g. AGENT-16's expiry/sunset drop).
  Tariff-threshold magic constant (`tariff_chunker.py:38`) folded into
  task.md as an optional 2-line addendum rather than its own task — trivial,
  but a different file/subsystem, so marked skippable if it would dilute
  review of the real fix (same "don't bundle unrelated concerns" discipline
  AGENT-14 used when it split the *original* AGENT-14 scope into 14 + 17).
- **AGENT-16 scoping (2026-08-31)**: the old plan's "AGENT-16" label
  ("amend/repeal/expiry lifecycle extraction") was one line covering
  three genuinely different-sized problems. Corpus-grounded before
  writing any regex: a precise operative-clause pattern (named act
  directly followed by `खारेज गरिएको छ`, no `को दफा`/`को उपदफा` in
  between) finds 157 documents with exactly one clean whole-act repeal
  declaration each — almost always inside a standard "खारेजी र बचाउ"
  दफा whose savings sub-clauses don't repeat the operative phrase, so
  they don't false-positive. A looser sweep (219 matches) shows the
  other 62 are a structurally different, harder pattern: partial repeal
  of one specific दफा of another act, and list-style दफाs mixing
  repeal with text-substitution consequential amendments — split out,
  not attempted here. `lifecycle_effect.effect_type` already allows
  `'repeal'` (migration 001, Phase 0) — **no schema change needed at
  all**, this is a pure extend-existing task per Ponytail. Separately
  grounded expiry/sunset-clause extraction: a broad `म्याद`/`कालावधि`
  keyword sweep returned 354 hits, all false positives on spot-check
  (generic "deadline" usage, not act-level sunset clauses) — **dropped,
  no evidenced pattern in this corpus**, not carried forward as a task.
  Clause-level `<amend>`-tag correlation against a document's own
  amendment-history table (the other third of the old "AGENT-16") is
  real and well-evidenced (15,748 tag instances across 519 documents,
  83% following one dominant `<Act name>, <year> द्वारा <verb>` shape)
  but comparably large on its own — renumbered **AGENT-18** below,
  not bundled in.
- **AGENT-15 scoping (2026-08-31)**: originally planned as "add a
  non-authoritative/unreviewed flag" on these three columns. Traced
  every downstream reader before implementing anything (per AGENTS.md
  "become one with the data"): `postgres_retriever.py::_hit()`,
  `validation_gate.py`'s `_citation()`/`_expression()`, and
  `eligibility_gate.py::eligible_chunk_ids()` — none of them select or
  reference `summary`/`keywords`/`relevant_questions` anywhere today.
  Ran the Ponytail gate against the literal "add a flag" plan: a new
  column with zero live consumers and no review workflow that could
  ever set it to a different value is exactly the speculative-field
  case the gate blocks. Rescoped to the smallest correct move instead:
  (A) `tests/test_metadata_enricher.py` — first-ever direct unit tests
  for `_parse_json`/`_apply_chunk_metadata`'s malformed-LLM-JSON
  handling (zero coverage existed); (B)/(C) regression tests pinning
  that these columns never reach the model context, a rendered
  citation, or the eligibility gate, so a future change that starts
  threading them through breaks a test loudly instead of silently
  weakening Core Invariant #8; (D) `COMMENT ON COLUMN` schema
  documentation only, no behavior change. No new column/table/
  abstraction added.
- **AGENT-14 scoping (2026-08-30)**: corpus-wide regex count against all
  677 `laws.jsonl` records confirmed AGENT-13's single-document finding
  is systemic, not a one-off: `_HEADER_RE`'s three loose (non-bold)
  alternatives match almost entirely inline cross-references, not real
  headers — दफा 10,766 total matches vs. 8 at true line-start; परिच्छेद
  3,270 vs. 7; धारा 651 vs. 0. Root cause of AGENT-13's 3,712
  duplicate-expression components. Also found while grounding: `parser
  .py::parse_law()`'s `source_sha256` doesn't NFC-normalize before
  hashing (unlike `pipeline.py::_content_hash()`), diverging for 2/677
  corpus records — breaks the `documents.content_hash` ↔
  `source_publication.source_sha256` provenance match PS-3 needs. Both
  bundled into AGENT-14 (same file, same investigation) plus a cleanup
  script for the already-corrupted live-DB rows. **Split out of the
  originally-planned AGENT-14 scope** (VALIDATE-stage structural
  hardening + tariff-threshold magic constant) into a new **AGENT-17**,
  below — bundling all of it risked an oversized, harder-to-review diff
  for two unrelated concerns.
- **DB check before scoping (2026-08-30)**: queried the live local DB
  directly rather than assume. `documents`/`work`: 345 rows each.
  `component`/`source_publication`/`expression`/`lifecycle_effect`: **0
  rows, all four**. AGENT-11's persistence and AGENT-12's commencement
  extraction only run inside `ingest_law()`, which the content_hash
  idempotency skip short-circuits before either stage for anything already
  ingested. Backfilling commencement alone (the literal AGENT-12 backlog
  ask) would need `source_publication` rows that don't exist either — so
  AGENT-13 backfills the whole authority layer in one script, not just
  commencement.
- Ingestion-pipeline gap analysis (2026-08-30): a pasted external-agent
  review of `app/ingestion/pipeline.py` was verified claim-by-claim against
  code. All 7 claims CONFIRMED — see AGENT-11 entry below for detail.
  AGENT-11 closes claims #1/#3 (component/source/expression persistence).
  Root cause was structural, not missing code: `writer.py` already had the
  needed functions; `pipeline.py` never wired them in.
- AGENT-12 scope narrowed after corpus grounding (2026-08-30): grepped all
  677 `laws.jsonl` records for commencement/repeal/amend phrasing before
  writing any regex (AGENTS.md "become one with the data"). Found: (a) four
  distinct commencement patterns (immediate / N-days-relative /
  gazette-dependent / नियमावली-own-publication), not two; (b) a naive
  "राजपत्रमा सूचना प्रकाशन गरी तोकिएको" regex is a false-positive trap —
  ~96 hits, almost all boilerplate definitions of "तोकिएको" unrelated to
  commencement, must anchor on the full clause ending in `प्रारम्भ हुनेछ`;
  (c) what looked like "repeal" mentions are actually entries in each
  document's amendment-history table (other acts that amended this one, by
  name/year) — not inline repeal instructions; real amend/repeal extraction
  needs correlating that table against `<amend>` tags, a harder, separate
  problem. Decision: AGENT-12 = commencement extraction only + a
  dual-approval review CLI (nothing in the codebase can approve *anything*
  yet — this CLI is the first). Amend/repeal/expiry moved to AGENT-15.
- **Honest scope note carried into AGENT-12's brief:** approving a
  commencement proposal does not yet change retrieval — `eligibility_gate.py`
  still derives eligibility from `documents`/`chunks` only, not from
  `component`/`lifecycle_effect`/`is_eligible()`. Rewiring the real gate to
  this bitemporal layer is a distinct future task.
- Not a task — informational finding only (2026-08-31, surfaced during
  AGENT-17's round-2 review, doesn't clear the bar for a numbered task: no
  Core Invariant / gate risk, not blocking, not requested). A small set of
  technical/tabular regulation schedules (weights & measures, customs
  tariff, engineering/health/education/insurance नियमावली) have अनुसूची
  schedules with their own internal sub-tables that each independently
  restart numbering from 1 (e.g. `स्टाण्डर्ड नाप र तौल नियमहरु २०२७` — 115
  components, `भन्सार_महसुल_ऐन_२०८१` — 69, still relying on AGENT-17's
  `/occurrence/N` disambiguation fallback rather than proper nested
  classification). Data integrity holds today (0 duplicate URIs either
  way) — this is a citation-precision nice-to-have for a narrow corpus
  slice, not a correctness gap. Only becomes a task if Prakash asks.
- Planned follow-on tasks (not yet branched): none currently — AGENT-18
  is now actively assigned (see Current task above, superseding the old
  backlog description below it used to sit under).

---


## Architecture decisions (standing)

### Authority weighting — Option A: Tier-first, RRF-second (decided 2026-08-24)
Sort retrieved chunks by work_type tier, break ties by RRF score. No blended weights.
Tier order: Constitution(1) > Act(2) > Rule/Regulation(3) > Directive/Byelaw(4) > Notification/Order(5) > Precedent(6).
Rationale: legally deterministic, auditable, no eval data needed to calibrate.
Upgrade path: move to weighted blend (Option B) once eval slice data justifies a specific α/β split.
Ref: `docs/adr-001-multi-agent-query-architecture.md` §Authority Weighting.

### Missing facts handling — Hybrid (decided 2026-08-24)
Fact extractor classifies each missing fact as: required | clarifying | informational.
- required    → interrupt graph, ask user before retrieving
- clarifying  → ask user if within wall-clock budget, else proceed and document
- informational → document in answer output, never blocks
Ref: `docs/adr-001-multi-agent-query-architecture.md` §Missing Facts.

---

---

## Completed tasks

### AGENT-18 — Correlate `<amend>` tags with amendment table into `lifecycle_effect` (MERGED to dev, 2026-08-31)
- `app/ingestion/amend_extractor.py` (new, deterministic, no LLM — same
  shape as `commencement_extractor.py`/`repeal_extractor.py`):
  `parse_amendment_table()` reads the ordered amendment table every
  document carries near its top (`संशोधन गर्ने ऐन` for acts, bare
  `संशोधन`/`संशोधन गर्ने नियम` for regulations) into `(position, name,
  bs_date)` entries, tolerant of act names that wrap across a line break
  (a `<=2`-digit guard on row-start markers distinguishes a new row from
  a continuation line, with a require-date finalize check running after
  every line so accumulation stops at the right boundary). `classify_
  amend_text()` resolves each `<amend>...</amend>` tag by either exact
  named-act match (reusing `enabling_extractor.py::_normalize_title()`
  against the document's own table, not the `work` table) or ordinal
  table-position (reusing `commencement_extractor.py::ORDINAL_DAYS`,
  same word→number mapping used as a 1-based position instead of a
  day-offset, with a `_normalize_ordinal()` pass unifying chandrabindু/
  अनुस्वार and other spelling variants before lookup — verified
  collision-free against all 41 dictionary entries). `_component_spans()`
  resolves each tag's enclosing दफा by offset, reusing `parser.py`'s
  `_HEADER_RE`/`_component_kind`/`_component`/`_disambiguate_component_
  uris` primitives directly (duplicates only the orchestration loop, not
  the matching logic — `parser.py` itself was off-limits this task).
  Anything that doesn't cleanly resolve is skipped and counted, never
  guessed at.
- `app/authority/writer.py::propose_lifecycle_amend()`: mirrors
  `propose_lifecycle_commence()` — always `approval_status='pending'`,
  empty `legal_valid_time` + a documented `amendment_date_unresolved:`
  dependency sentinel when the table's date doesn't resolve (no
  fabricated date), dedup on `(component_uri, effect_type='amend',
  approval_status='pending', raw_clause_text)`.
- `app/ingestion/pipeline.py`: wired into the existing `PROPOSE_LIFECYCLE`
  span (AGENT-12, extended by AGENT-16/this task), same
  `SAVEPOINT`/`ROLLBACK TO SAVEPOINT` best-effort discipline — not a new
  stage. `effect_type='amend'` and `EffectType.AMEND` already existed
  (migration 001, `models.py`) — no schema change.
- **Corrected grounding before implementation**: the backlog note carried
  from AGENT-16's scoping claimed "83% dominant named-act pattern,
  ordinal secondary" — a precise per-tag classification of all 15,748
  instances found the opposite (named-act 31.3%, ordinal-position 55.9%,
  the actual majority) before any code was written — see AGENT-18
  scoping note above for full detail.
- **Two-round review, both independently re-verified against the live
  corpus** (Claude Review Gate) — see Status notes above for full detail.
  Round 1 (`e73103a`) correctly resolved 6,424 tags with sound dedup and
  abstention design (hand-verified against real documents), but two skip
  buckets hid fixable causes: a table-row regex silently dropping any
  row whose act name wrapped a line (68/498 docs), and an ordinal lookup
  that didn't normalize spelling variants (936 tags, 55 distinct
  tokens). Round 2 (`d99712a`) fixed both, hand-traced against the real
  documents that surfaced them, zero cross-number collisions confirmed
  in the normalization table.
- **Final corpus dry-run**: 451/519 amend-tag-bearing documents produce
  at least one proposal; 7,358 total proposals (1,941 named-act, 5,417
  ordinal); 8,390 tags skipped and accounted for (4,492 duplicate — same
  दफा + same table entry + identical tag text, verified by hand as
  redundant markers of one edit event, not lost facts; 3,141
  unresolved_other — explicitly out-of-scope content per the task's own
  grounding; 570 unresolved_named-act — act genuinely absent from that
  document's table, correctly abstained; 187 no_table — no discoverable
  amendment table at all). 161 tests passing (3 skipped), lint clean,
  eval-gates all at 0.
- **Honest scope note (explicit per task brief, still true)**: this
  records amendment *facts* only (which दफा, by which act, when) — this
  corpus is a single current-snapshot with no pre-amendment text to
  version, so it does not achieve full PS-17 compliance (closing a prior
  `expression`'s `valid_time` at the retroactive date). Same lineage as
  every other lifecycle-extraction task's "doesn't wire into retrieval
  yet" caveat (AGENT-11/12/13/16): this populates the authority store
  with real facts, it doesn't change what retrieval serves.
- **Known minor gap (not blocking)**: `unresolved_other` (3,141 tags,
  ~20% of the corpus total) was explicitly scoped out per the task
  brief's own grounding — commencement dates, repeal asides, name
  changes, and unrelated content already covered elsewhere or genuinely
  out of scope. Not revisited this task; would need its own grounding
  pass if ever pursued, and per [[feedback_task_creation_bar]] only
  becomes a task if Prakash asks.

### AGENT-17 — Fix दफा component-URI collisions (schedule + compound numbering) (MERGED to dev, 2026-08-31)
- `app/authority/parser.py`: `_HEADER_RE` gained an अनुसूची (schedule) boundary
  alternative (`अनुसूची\s*[-–]?\s*N`, tolerant of the corpus's inconsistent
  dash/whitespace formatting) so schedule content stops being misclassified
  as `dafa` components; bold-दफा numbers now optionally capture one compound
  `.M` decimal level, with the punctuation check widened to also accept
  "number, whitespace, free-form title, colon before the closing `**`" (the
  real format for this corpus's compound headers, which don't have
  punctuation immediately after the number). `_component_kind()` threads a
  `schedule_number` loop variable so items after an अनुसूची boundary get
  `component_type="anushuchi"` with a `{schedule}.{item}` URI instead of
  colliding with real दफा numbers. New `_disambiguate_component_uris()` is
  an unconditional final pass over every parsed document — any URI that
  still collides after reclassification (genuine same-number,
  different-content दफा pairs in the source text itself, not decidable as
  "which one is correctly numbered") gets a deterministic `/occurrence/N`
  suffix so both provisions stay distinctly addressable; nothing is ever
  silently dropped or overwritten.
- `app/ingestion/pipeline.py`: VALIDATE stage (laws path) now asserts
  `parse_law()`'s output has no duplicate component URIs and rejects if it
  does — a regression safety net for any future numbering pattern not in
  today's corpus, not a substitute for the parser fix (by construction,
  `_disambiguate_component_uris` already guarantees this can't fire today;
  the check guards against a future parser change silently reintroducing
  the invariant violation).
- `scripts/cleanup_stale_authority_expressions.py`: `_expected_hashes()` now
  raises loudly on a duplicate URI instead of silently keeping only the
  first (dead code today, same defense-in-depth reasoning as the VALIDATE
  guard); new `_missing_component_uris()`/`_upsert_missing_components()`
  insert `component` rows for the newly-introduced disambiguated URIs
  (`anushuchi/...`, `.../occurrence/N`) that didn't exist under the old
  parser; existing orphan-cleanup + lifecycle-status-guard logic
  (AGENT-14) reconciles the old, now-superseded URIs unchanged — reused,
  not reimplemented, per Ponytail.
- **Two-round review, both independently re-verified against the live
  corpus rather than trusting the engineer's report** (Claude Review
  Gate) — see Status notes above for full detail. Round 1 (`cf3c1aa`)
  claimed the fix but the compound-number and अनुसूची-boundary fixes didn't
  actually fire on the task's own grounding examples, only the
  disambiguation net was doing the work; sent back with exact regex
  diagnosis (`task.md` rework note, `d10496e`). Round 2 (`461fcb9`) fixed
  both, verified against the same real documents (आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५
  now parses cleanly to `2.1`...`2.9`; `स्टाण्डर्ड नाप र तौल नियमहरु २०२७` now
  recognizes 185 अनुसूची components, was ~0), new tests load the real
  corpus record via `laws.jsonl` instead of a synthetic string that didn't
  match the actual format.
- 0/677 documents with duplicate component URIs (was 61/677, 857 excess
  rows), independently re-derived, not just re-quoted from the report.
  153 tests passing (3 skipped), lint clean, `make eval-gates` all three
  zero-tolerance gates at 0. The 5 mypy errors on the 3 ingestion-path
  files not in the Makefile's fixed lint list are byte-for-byte
  pre-existing on the base, confirmed by diffing mypy output before/after
  on the unmodified files.
- **Honest scope note**: a deeper, structurally different pattern surfaced
  during round-2 review — a handful of technical/tabular regulation
  schedules (weights & measures, customs tariff, engineering/health/
  education/insurance नियमावली) have schedules with their own internal
  recursive numbering restarts, still relying on the `/occurrence/N`
  safety net rather than proper nested classification (590 components /
  78 documents, dominated by ~5 technical schedules). Data integrity holds
  (0 duplicate URIs either way) — this is a citation-precision gap for a
  narrow technical-schedule corpus slice, not a correctness risk, and it's
  outside what this task's grounding scoped. Not sent for a third rework
  round, and not given a task number — no Core Invariant/gate risk, not
  requested — logged as an informational finding in the backlog section
  above instead of a planned task.
- Live-DB cleanup script (`--dry-run` then real run) could not be executed
  this session — local Postgres was not running in either engineer's
  environment. **Still pending on Prakash**, same operational-steps
  pattern as other backfill/cleanup scripts in this file (see "Operational
  steps still pending on Prakash" below) — run
  `python3 scripts/cleanup_stale_authority_expressions.py --dry-run` then
  for real against the live local DB, and report before/after counts the
  same way AGENT-14's cleanup run was reported.

### AGENT-16 — Extract whole-act repeal declarations into lifecycle_effect (MERGED to dev, 2026-08-31)
- `app/ingestion/repeal_extractor.py` (new): deterministic, no LLM — `_REPEAL_RE` matches a named act/ordinance/regulation directly followed by `खारेज गरिएको छ` (handles an optional bold दफा-header prefix and an optional `(१)` sub-clause marker before the title). `classify_repeal()` returns a 3-outcome `RepealMatch` (`auto_extracted` / `repealed_work_not_in_corpus` / `no_repeal_clause`), reusing `enabling_extractor.py`'s `_normalize_title`/`_resolve_work` unchanged (same name-to-`work`-row problem, no reimplementation). Corpus-count regression test hard-asserts `(157, 157)` docs/matches, confirmed independently during review. Verified the savings sub-clauses (e.g. `(२) ... बमोजिम भए गरेका काम ... मानिनेछ`) and the partial-दफा-of-another-act pattern (e.g. `"...ऐन, YYYY को दफा N ... खारेज गरिएको छ"`) both correctly produce `no_repeal_clause`, not a false match — tested explicitly.
- `app/authority/writer.py::propose_lifecycle_repeal()`: fans out one pending `effect_type='repeal'` proposal per `component.uri` belonging to the repealed work (`lifecycle_effect.effect_type` already allowed `'repeal'` since migration 001 — no schema change). Repeal date resolution: looks up `MIN(effective_date)` across the *repealing* work's own approved `commence` effects; if found, uses it and clears the dependency; if not, `legal_valid_time='empty'` + `commencement_dependency='repealing_work_commencement:<uri>'` — no fabricated date (PS-2), mirrors AGENT-12's `gazette_notification_pending` honesty pattern. Dedups per-component on `(component_uri, effect_type='repeal', approval_status='pending')`, same discipline as `propose_lifecycle_commence`.
- `app/ingestion/pipeline.py`: wired into the existing `PROPOSE_LIFECYCLE` span (added by AGENT-12) alongside commencement extraction, same `SAVEPOINT`/`ROLLBACK TO SAVEPOINT` best-effort discipline — not a new stage.
- `scripts/review_lifecycle.py`: generalized the hardcoded `effect_type='commence'` filters into an `--effect-type` flag (`list`/`list_work`/`approve_work`/`reject_work` all accept it; omitted shows all types) — repeal proposals are now listable/approvable through the same dual-sign-off CLI. Without this fix they would have sat `pending` forever, undiscoverable, silently defeating Core Invariant #5's human-gating even though the row itself stayed technically ungranted.
- 147 tests passing (139 + 8 new), lint clean. `app/ingestion/` still isn't in `make lint`'s fixed file list (same pre-existing gap noted in AGENT-14/15) — checked `repeal_extractor.py`/`pipeline.py` manually: the only new-looking error (`repeal_extractor.py:36`, "Returning Any") was confirmed to be a pure `--follow-imports=skip` artifact (disappears when mypy is allowed to follow the `enabling_extractor` import and see `_normalize_title`'s real `-> str` signature); `pipeline.py`'s 4 errors are byte-for-byte pre-existing on the unmodified base file. Zero new lint/type issues from this diff. Eval-gates all at 0 — `repealed-as-current` is (as documented in the honest scope note below) a synthetic self-test unaffected by real corpus data either way, confirmed unchanged.
- **Corpus result**: 157 documents' whole-act repeal clauses are now extractable; running ingestion against the live corpus will queue pending repeal proposals for review through `review_lifecycle.py --list --effect-type repeal`.
- **Honest scope note (still true after this task, same lineage as AGENT-11/12/13/15's commencement caveat)**: this populates the authority store with real repeal facts. It does **not** make retrieval respect them — `eligibility_gate.py::eligible_chunk_ids()` (what retrieval actually calls) still derives eligibility from `documents.ingestion_status`/`chunks.effective_date_ad` only, never `lifecycle_effect`. Rewiring that is a distinct future task.
- **Known minor gap (not blocking, flagged for a future observability pass)**: `classify_repeal`'s `repealed_work_not_in_corpus`/`no_repeal_clause` outcomes are returned in-memory but never persisted anywhere (unlike `enabling_extractor.py`, which always writes an audit-trail row to `work_relations` regardless of outcome) — `work_relations` reuse was explicitly rejected in this task's brief as a semantic mismatch, and no alternative persistence target was specified, so an unresolved repeal match (a repeal clause naming an act not yet in the corpus) currently leaves no queryable trace. Recoverable via re-parsing `laws.jsonl` from scratch if ever needed (same recourse as other backfill scripts), not silently lost forever — but not proactively discoverable today either.
- **Known minor limitation**: `_REPEAL_RE.search()` only extracts the first match per document. Corpus-verified as a non-issue today (all 157 matching documents have exactly one clean whole-act repeal clause, zero with two or more) — would need switching to `finditer()` if a future corpus document ever repeals two acts via two separate clean clauses.

### AGENT-15 — Regression-guard LLM-derived chunk metadata as non-authoritative (MERGED to dev, 2026-08-31)
- Rescoped from the originally-planned "add a non-authoritative/unreviewed flag" column after tracing every downstream reader of `documents.summary`/`chunks.keywords`/`chunks.relevant_questions`: none of `postgres_retriever.py::_hit()`, `validation_gate.py`'s `_citation()`/`_expression()`, or `eligibility_gate.py::eligible_chunk_ids()` read these columns today. Ponytail-blocked the literal flag plan (zero consumers, no review workflow that could ever flip it — a speculative field). See scoping note above for full reasoning.
- `tests/test_metadata_enricher.py` (new): first direct unit tests for `metadata_enricher.py::_parse_json`/`_apply_chunk_metadata` — malformed JSON, non-list JSON, markdown-fenced JSON, missing/wrong-typed `chunk_index`, wrong-shaped `keywords`/`relevant_questions` values, partial-batch-failure isolation. Zero coverage existed before this task.
- `app/ingestion/metadata_enricher.py`: `_apply_chunk_metadata`'s `by_index` construction previously did `int(item["chunk_index"])` unguarded outside the JSON-parse try/except — a non-int-convertible `chunk_index` (e.g. a non-numeric string) from a malformed LLM response would raise uncaught, crashing metadata enrichment. Found while writing part A's own test cases, not a separate investigation. Fixed to `isinstance(chunk_index, int)` + skip on mismatch — matches the module's stated "never guesses metadata" design; a numeric-looking string is no longer silently coerced.
- `tests/test_retrieval.py`: `test_hit_does_not_surface_llm_metadata` pins `_hit()`'s exact returned key set even when fed a wider input row; `test_retriever_sql_does_not_select_llm_metadata` / `test_validation_gate_sql_does_not_read_llm_metadata` assert (via `inspect.getsource`) that no SQL literal in `postgres_retriever.retrieve_postgres` or the `validation_gate` module names `summary`/`keywords`/`relevant_questions` — pins Core Invariant #8 (all retrieved text is untrusted) so a future change threading these columns into the model context or a citation breaks a test instead of landing silently.
- `tests/test_eligibility_gate.py`: same SQL-literal-absence assertion for `eligibility_gate.eligible_chunk_ids` — pins that unreviewed LLM metadata can never influence temporal eligibility.
- `migrations/005_ingestion_pipeline.sql` / `006_add_summary.sql`: `COMMENT ON COLUMN` on `chunks.keywords`/`chunks.relevant_questions`/`documents.summary` — LLM-derived, never human-reviewed, not authoritative, must never be rendered as or substituted for statutory text or a citation. Documentation only, no behavior change.
- `migrations/010_metadata_provenance_comments.sql` (new) + `scripts/migrate.py` registration: applies the same column comments to databases that already ran 005/006 — a judgment call made after checking `scripts/migrate.py` tracks migrations by name, not content-hash, so editing 005/006 alone wouldn't reach already-migrated DBs.
- 139 tests passing (3 skipped, pre-existing/unrelated), lint clean via `make lint`. `app/ingestion/` isn't in `make lint`'s fixed file list (pre-existing gap, same as noted in AGENT-14) — checked `metadata_enricher.py` manually with ruff + mypy --strict: the 2 `Missing type parameters for generic type "dict"` errors are pre-existing on the unmodified base file (verified by diffing mypy output before/after), not introduced by this change. Eval-gates all at 0.
- **Process note**: the engineer's diff was never committed — found sitting uncommitted in the working tree when review started. Root cause: `.gitignore` has a bare `tests` line (pre-dating most currently-tracked test files, which stay tracked despite it) that silently blocks `git add` on new files under `tests/` without `-f`; the new `tests/test_metadata_enricher.py` hit this. Verified the full diff content and all required checks independently before committing it myself (author `Prakash Basnet`, per policy) — content was correct, only the commit step was missed.
- **Hygiene note (not part of this task's diff)**: 3 untracked files were sitting in the working tree unrelated to this task — `.agent/extract_meta_bottleneck.md`, `docs/legal_rag_ingestion_best_practices.md`, `task.md.bak`. Left untouched (not staged, not part of the commit); flagged to Prakash for cleanup, origin unconfirmed.

### AGENT-14 — Fix दफा/परिच्छेद/धारा header over-matching + canonicalize source_sha256 (MERGED to dev, 2026-08-30)
- `app/authority/parser.py`: `_HEADER_RE`'s three loose (non-bold) alternatives (दफा, परिच्छेद, धारा) anchored to line-start — they were matching inline cross-references anywhere in a document's body (e.g. "यस ऐनको दफा ३ बमोजिम"), not just genuine headers. Corpus-wide verification before/after: दफा 10,743→0 matches, परिच्छेद 3,270→4 (all 4 confirmed genuine chapter headers, e.g. "परिच्छेद-१\nप्रारम्भिक"), धारा 649→0. Bold-header alternative untouched (26,622 matches, unaffected). Root cause of AGENT-13's finding (3,712 components with duplicate `expression` rows, one with 33). `source_sha256` in `parse_law()` now NFC-normalizes before hashing, matching `pipeline.py::_content_hash()` — previously diverged for 2/677 corpus records, breaking the `documents.content_hash` ↔ `source_publication.sha256` provenance match PS-3 needs.
- `scripts/cleanup_stale_authority_expressions.py` (new): two cleanup passes against the live DB, re-deriving expected state from the fixed `parse_law()` per document (same per-document commit/rollback + `--dry-run` discipline as `backfill_authority_layer.py`). Pass 1: deletes `expression` rows whose `text_hash` no longer matches the fixed parser's output for their `component_uri` (leftover fragments from partially-duplicated components). Pass 2 (added in rework round 1, see below): deletes fully orphaned `component`/`expression`/`lifecycle_effect` rows for section numbers that were **never** real headers at all — these don't show up in pass 1 because the URI itself disappears from the fixed parser's output, not just its fragment count. Also corrects stale `source_publication.sha256` values.
- **Rework round 1**: first-pass diff only reconciled URIs still present in the corrected parse, silently leaving fully-orphaned URIs untouched. Caught by an independent live-DB query during review (not just corpus regex counts): 1,901 orphan `component` rows, 2,350 orphan `expression` rows, 1,238 `lifecycle_effect` rows (all `approval_status='pending'`, none `approved` — verified before allowing deletion). Fix adds a lifecycle-status guard: an orphan URI with any non-`pending` lifecycle row is blocked and reported, never deleted (Core Invariant #5 territory — an approved lifecycle fact needs a human decision, not a script).
- `tests/test_parser.py`: cross-reference false-positive regression + non-NFC `source_sha256` regression. `tests/test_cleanup_stale_authority_expressions.py` (new): orphan cleanup counts + approved-row blocking, via a `FakeConn`/`FakeCursor` harness.
- 126 tests passing, lint clean (ruff + mypy --strict, including the new script which isn't in the Makefile's fixed lint file list — checked manually), eval-gates all at 0.
- **Run against the live local DB**: `component` 16,459→14,558, `expression` 23,187→14,601, `lifecycle_effect` 10,619→9,381. Independently re-verified post-merge: zero remaining orphans, zero non-pending-lifecycle violations, table counts match exactly.
- **Scope note**: originally planned as part of a broader AGENT-14 (VALIDATE-stage hardening + tariff constant bundled in) — split during scoping once corpus grounding turned this into a concrete, evidenced correctness bug on its own. The broader hardening work is now **AGENT-17**.

### AGENT-13 — Backfill authority layer for already-ingested laws (MERGED to dev, 2026-08-30)
- `scripts/backfill_authority_layer.py` (new): sources records from `laws.jsonl` matched by `source_id` (not reconstructed from `documents` columns — `documents` doesn't store `name`/`work_id`, checked against `migrations/005_ingestion_pipeline.sql` before writing anything), verifies `_content_hash(record["content"]) == documents.content_hash`, asserts `parse_law(record).uri == work.uri` (fetched via `chunks.work_id` join, same pattern as AGENT-10's `backfill_enabling_links.py`) before writing anything — avoids silently computing a different `component.uri` than what `pipeline.py::_commence_date()` would ever query for. Per-document commit/rollback (not one giant transaction); a mid-write failure on one document doesn't lose prior documents' committed work. `--dry-run` flag.
- Replays AGENT-11's `upsert_source`/`upsert_component`/`upsert_expression` + AGENT-12's `extract_commencement_proposals` unmodified — no new writer logic, this task only orchestrates existing functions
- `tests/test_backfill_authority_layer.py`: 8 tests via a `FakeConn` harness with real snapshot/rollback semantics — happy path, missing record, hash mismatch, missing work_id, uri mismatch, idempotency (second run writes 0 new rows), dry-run, mid-write failure rolls back cleanly and continues to the next document
- 122 tests passing, lint clean, eval-gates all at 0
- **Run against the live local DB** (not just tests): 345/345 documents backfilled, zero skips of any kind (no missing records, no hash mismatches, no missing work_ids, the `law.uri != work.uri` assert never fired), idempotency confirmed by a second real run writing 0 new rows. Final counts: `component` 16459, `source_publication` 345, `expression` 23187, `lifecycle_effect` 10619. Lifecycle breakdown: 9279 resolved commence rows, 1210 `gazette_notification_pending`, 95 `no_commencement_clause`, 35 `enactment_date_unknown`.
- **Found during review, not a defect in this task** (see AGENT-14 above for the concrete evidence): `expression` count exceeds `component` count because `parser.py`'s दफा-header regex matches the same section number more than once in some documents — pre-existing, out of this task's scope by its own brief (parser.py was explicitly off-limits).
- **Honest scope note (still true)**: this makes the bitemporal store populated, not the retrieval gate temporal-correct — `eligibility_gate.py` rewiring is still a separate future task.

### AGENT-12 — Commencement proposal extraction + dual-approval review CLI (MERGED to dev, 2026-08-30)
- Corpus-grounded before writing any regex — see scope-narrowing note above (four commencement patterns, gazette-notification false-positive trap avoided, amend/repeal split to AGENT-15)
- `migrations/009_lifecycle_raw_clause.sql`: `lifecycle_effect.raw_clause_text TEXT` (audit trail, same pattern as `work_relations.raw_clause_text` from AGENT-10); registered in `scripts/migrate.py`
- `app/authority/writer.py`: `propose_lifecycle_commence()` — always writes `approval_status='pending'`; empty-range `legal_valid_time` (`'empty'::tstzrange`) when `effective_date` is unknown rather than an unbounded range that would silently assert always-valid; dedup on `(component_uri, effect_type='commence', approval_status='pending')`; `insert_commence`'s Phase-0 auto-approve stub left untouched and unused
- `app/ingestion/commencement_extractor.py` (new): four-pattern regex classifier (immediate / N-days-relative / gazette-dependent / नियमावली-own-publication), 41-entry Devanagari-ordinal→day table with safe abstention (`unparsed_relative_delay`) on unrecognized words rather than guessing; one proposal per दफा component for real matches (matches `pipeline.py::_commence_date()`'s per-दफा lookup), exactly one proposal keyed to `law.uri` for the `no_commencement_clause` sentinel (not fanned out per-दफा — fixed in review, see below)
- `app/ingestion/pipeline.py`: new `PROPOSE_LIFECYCLE` span after `PERSIST_AUTHORITY`/before `CHUNK`; wrapped in `SAVEPOINT`/`ROLLBACK TO SAVEPOINT` (not a bare try/except) so a failure can't poison the rest of `ingest_law()`'s transaction — best-effort, does not reject the document (proposals aren't authority data); `PERSIST_AUTHORITY`'s `upsert_source()` return value now captured and threaded through as `source_pub_id`
- `scripts/review_lifecycle.py` (new): `--list`/`--list-work`, `--approve`/`--reject <id> --by <uuid>`, `--approve-work`/`--reject-work <work_uri> --by <uuid>` (bulk convenience over the fan-out problem, flagged as a deliberate addition beyond the literal spec). Dual sign-off enforced by SQL logic: `FOR UPDATE` row lock, first approver recorded without flipping status, a second *distinct* approver required to reach `'approved'`, same-person double-approval refused via `str(approver1) == by`. Reject is single-action (lower-risk direction), reuses `approved_by_1` for rejecter attribution. `no_overlap` EXCLUDE constraint violations caught and reported cleanly, not as a raw traceback. First tool in this codebase that can approve anything (document-level dual approval has no CLI either, still raw SQL only).
- `tests/test_commencement_extractor.py` (10 tests), `tests/test_review_lifecycle.py` (7 tests) — 114 tests passing total, lint clean, eval-gates all at 0
- Review fixes (commit 826b424, before merge): (1) `no_commencement_clause` sentinel was fanning out one row per दफा component (171/677 no-match docs × avg दफा count — thousands of redundant rows, `--list` clutter); fixed to write exactly one row keyed to `law.uri`. (2) immediate-commencement branch silently dropped `commencement_dependency` when `law.enactment_ad` was `None`, unlike the other three branches; fixed to set `'enactment_date_unknown'`.
- **Honest scope note (carried forward, still true):** approving a proposal does not yet change retrieval — `eligibility_gate.py` still derives eligibility from `documents`/`chunks` only, not `component`/`lifecycle_effect`/`is_eligible()`. That rewiring is a separate future task.
- **Backlog note (Pi's return, unresolved):** re-running `scripts/ingest_laws.py` today will NOT create commencement proposals for the 677 already-ingested laws — the pre-trace `content_hash` idempotency skip returns before `PROPOSE_LIFECYCLE` ever runs. No backfill script was written this task (deliberately, per brief). Prakash needs to decide: a one-off backfill pass (separate small task), or accept lifecycle proposals only apply to laws ingested/changed going forward.
- Corpus pattern counts (all 677 laws): immediate 460, relative-days 5, gazette-dependent 40, नियमावली-own-publication 1, no_commencement_clause 171. Distinct ordinal words actually found in corpus: 4 (आठौं, एकतिसौँ, एकतीसौँ, एकानब्बेऔं) — all covered by the table.

### AGENT-11 — Persist parsed law authority structure (MERGED to dev, 2026-08-30)
- `app/ingestion/pipeline.py`: new `PERSIST_AUTHORITY` span in `ingest_law()`, after VALIDATE and before CHUNK — calls `upsert_source`, then `upsert_component`/`upsert_expression` per parsed component (`as_of=date.today()`, computed once); any failure rejects the document (`_set_status(..., "rejected")`, matches CHUNK-stage rejection shape) rather than silently continuing — components/expressions are authority data, not a derivative annotation
- `tests/test_ingestion_pipeline.py`: 6 new tests — component/source/expression persistence, `as_of` correctness, skip-path (no persistence calls on unchanged content_hash), persistence-failure rejects + never reaches chunker, VALIDATE-failure precedes persistence; `test_langfuse_span_end_called` updated for the new stage
- 95 tests passing, lint clean (ruff + mypy --strict), eval-gates all at 0
- Review fix: initial diff (dde4267) put `"error": str(exc)` into the `PERSIST_AUTHORITY` span's rejected-branch output — the only stage in the file to put raw exception text into a Langfuse span. Risk: `upsert_expression` inserts `text_ne` (full दफा text) as a column; a NOT NULL violation on that insert surfaces Postgres's `DETAIL: Failing row contains (...)` — full row values — inside `str(exc)`, which would then leak into the trace (PS-14). Fixed in 1411b79: `"error_type": type(exc).__name__` instead, with a test asserting the span output carries no raw `"error"` key.
- Scope explicitly excluded lifecycle proposal extraction and any call to `insert_commence` (Phase-0 auto-approve stub) — see AGENT-12 below
- Deferred to follow-on tasks (not fixed here): shallow VALIDATE stage, `content_hash` canonicalization, tariff-routing magic threshold, LLM-derived metadata authoritativeness flag — see AGENT-13/14 in Status above

### AGENT-10 — Enabling-power links (MERGED to dev, 2026-08-30)
- `migrations/008_work_relations.sql`: `law_level` enum widened (tariff_heading/row/note, unblocks AGENT-9 ingest) + `work_relations` table with section-aware unique indexes and `valid_time` (PS-6)
- `app/ingestion/enabling_extractor.py` (new): two-regex extraction (strict + उपदफा variant), amend-markup strip, comma-normalize for `work.title_ne` resolution, explicit `no_enabling_clause` sentinel rows — no LLM, fully deterministic
- `app/ingestion/pipeline.py`: `_NIYAM_RE` suffix check triggers extractor after CHUNK stage; exception guard so failure never blocks ingest
- `app/retrieval/query_graph.py`: `_fetch_enabling_chunk` + `enabling_power_resolver_node` inserted after `cross_ref_resolver`; eligibility gate mandatory (PS-6); deduplication of parent chunks; Langfuse span for observability
- `scripts/backfill_enabling_links.py` (new): idempotent post-processing for already-ingested नियमावली
- `scripts/migrate.py`: `008_work_relations` entry + idempotency probe
- `tests/test_enabling_extractor.py` (new): 9 tests — standard regex, उपदफा variant, amend markup, normalization, resolved/unresolved/no-clause insertion, idempotency
- `tests/test_enabling_retrieval.py` (new): 4 tests — eligibility gate respected, happy-path co-retrieval, null link skipped, duplicate parent deduplication
- 89 tests passing, lint clean (mypy clean), eval-gates all at 0
- Review fix: Kimi removed pre-existing `# type: ignore[import-not-found]` on langfuse import in `postgres_retriever.py`; restored by Claude in fixup commit
- `scripts/backfill_enabling_links.py` fix (2026-08-30, on dev): script failed on standalone execution (`ModuleNotFoundError: app`) — added repo-root `sys.path` insertion so `python scripts/backfill_enabling_links.py` resolves `app.ingestion.enabling_extractor` without requiring `PYTHONPATH`. Prakash ran it successfully against local DB — carry-forward resolved.

### AGENT-9 — TariffChunker + detection gate (MERGED to dev, 2026-08-29)
- `app/ingestion/tariff_chunker.py` (new): `is_tariff_dominant()` (>5000 HS codes + tariff keyword), `TariffChunk` dataclass (identical fields to `LawChunk`), `TariffChunker.chunk_text()` — parses pipe-table rows into `tariff_heading` / `tariff_row` / `tariff_note` chunks with deterministic keywords, `embed_text` from structured fields, and `co_retrieve_parent_index` linkage (PS-16)
- `app/ingestion/pipeline.py`: routing condition at CHUNK stage — tariff-dominant content → `TariffChunker`, skips `enrich_law_chunks`, EXTRACT_METADATA span emitted with `llm_calls=0`
- `app/ingestion/pgvector_indexer.py`: `TariffChunk` import + `isinstance` branch for PS-10-correct `chunk_type` (`"tariff_heading"` / `"tariff_row"`)
- `app/retrieval/postgres_retriever.py`: pre-existing mypy `type: ignore` added (1-line; fixes AGENT-8 carry-forward)
- `tests/test_tariff_chunker.py` (new): 7 tests — detection gate, heading/row linkage, embed_text richness, deterministic questions, pipeline routing
- Verification: `भन्सार_महसुल_ऐन_२०८१` → 10,381 chunks (1,309 headings, 5,265 rows, 3,807 notes) vs. 892 broken prose chunks before
- 76 tests passing, lint clean, eval-gates all at 0
- Carry-forwards: type annotations on `upsert_document`/`_chunk_row` missing `TariffChunk` (runtime-correct, mypy doesn't cover ingestion); chapter title not captured in embed_text (chapter number present)

### AGENT-8 — Production-grade ingestion observability (MERGED to dev, 2026-08-26)
- `pipeline.py`: `_span()` replaced with `_begin_span()` / `_end_span()` / `_end_trace()` — every span now has non-null `endTime`; every stage has `input`/`output` fields; per-stage stdout with `flush=True`; root trace updated with totals and `.end()` called on all paths including skipped/rejected/quarantined
- `metadata_enricher.py`: `_call_llm()` returns `(content, usage)` tuple; creates a Langfuse **generation** per LLM call with `model`, PS-14-gated `input`/`output`, and `usage_details` from `response.response_metadata["token_usage"]`; `_parallel_chunk_metadata` / `enrich_law_chunks` / `enrich_nkp_chunks` return 4-tuples `(metadata, llm_calls, in_tok, out_tok)`
- `pgvector_indexer.py`: `embed_chunks()` returns `(embeddings, total_tokens)`; creates embedding **generation** with `usage_details` from `response.usage.total_tokens`
- `tests/test_ingestion_pipeline.py`: 2 new tests (`test_langfuse_span_end_called`, `test_enrich_law_llm_call_count`); 69 total passing
- Lint: ruff clean; mypy pre-existing failure in `postgres_retriever.py` (import-not-found: langfuse) unrelated to task scope

### AGENT-7 — Unified Langfuse trace (MERGED to dev, 2026-08-26)
- `postgres_retriever.py`: `_get_lf_client` → `get_lf_client` (exported); `retrieve_postgres(lf_trace=None)` — creates `retrieval_span` as child of `lf_trace` instead of root trace; all `lf.trace()` and `lf.flush()` calls removed; `_end_span` now in try/except; `retrieval_span.end(metadata=...)` at normal return with `eligible_count`, `final_count`, `top_vector_score`
- `gated_orchestrator.py`: `_langfuse_callback(trace_id=None)` — passes `trace_id` to handler to link LLM generations as children; `_fact_extract`, `_structured_claims`, `_compose_answer` all gain `lf_trace=None` param; `_emit_answer_trace` removed; `_emit_answer_trace_from_state` rewritten — takes `lf_trace` as first arg, calls `lf_trace.update(output=...)` + `lf_trace.end()`; `import hashlib` removed (moved to query_graph)
- `query_graph.py`: `run_query` creates root `rag.query` trace via `get_lf_client()`, passes via `config["configurable"]["lf_trace"]`, flushes with `_lf.flush()` after invoke; every node pulls `lf_trace` from config; `authority_ranker_node`, `cross_ref_resolver_node`, `validate_node` each create a child span; `answer_composer_node` calls `_compose_answer` first (generation fires), then `_emit_answer_trace_from_state` (root trace ends); interrupted path also ends root trace
- `tests/test_orchestrator.py`: `test_emit_trace_uses_vector_score` updated — uses `FakeTrace` object with `.update()/.end()`, confirms vector_score used and gate_decision correct
- Minor smell (carry forward): redundant `if lf_trace is not None else` ternary in 4 node functions — functionally correct (default is None), can be simplified in cleanup pass

### AGENT-6 — Observability Fix (MERGED to dev, 2026-08-25)
- `postgres_retriever.py`: `_span` → `_end_span` — calls `span.end()` so all spans have `endTime`; `_hit()` gains `vector_score` param; `vector_scores` dict built from vector search rows; cosine similarity propagated through RRF and rerank to returned hits
- `gated_orchestrator.py`: `_langfuse_callback()` now used on all 3 LLM calls (`_structured_claims`, `_fact_extract`, `_compose_answer`); explicit `callbacks[0].langfuse.flush()` after each invoke; `_emit_answer_trace_from_state` uses `vector_score` (not RRF score) for `top_chunk_scores`; `LANGFUSE_LOG_CONTENT` flag gates raw `query` + `answer_summary` fields in trace; `_compose_answer` strips markdown code fences before `json.loads`
- `config.py`: `LANGFUSE_LOG_CONTENT: bool = False` added
- `tests/test_orchestrator.py`: `FakeResp.content` in compose test now uses markdown-wrapped JSON to verify fence stripping; `test_emit_trace_uses_vector_score` added; 67 total passing
- PS-14 maintained: raw content off by default; latent risk noted — `_fact_extract` JSON parse lacks fence stripping (can add in cleanup pass)

### AGENT-5 — Answer Composer + Missing-Facts Interrupt (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_classify_and_decompose` removed (dead since AGENT-2); `import os` removed; `_compose_answer(facts, missing_facts, all_results, conflict_hits, session_as_of)` added — Gemini 2.5 Flash composes ADR Node 7 format (`relevant_sections`, `plain_language`, `missing_facts`, `conflicts`, `disclaimer`); filters to clarifying/informational missing facts only; `except Exception: return None` fallback
- `query_graph.py`: `fact_extractor_node` detects `required` missing facts → sets `interrupted=True` + `interrupt_prompt`; `assemble_node` replaced by `answer_composer_node` — interrupt short-circuit (returns directly, bypasses retrieval) or normal path (Gemini compose + fallback to raw claims); `build_graph()` uses `add_conditional_edges` from `fact_extractor` → `answer_composer` (interrupt) or `retrieve` (normal); graph still 7 nodes, one of which is now reached via two paths
- `tests/test_orchestrator.py`: `test_classifier_failure_falls_back_to_simple` removed; 3 new tests added (compose success, no-key fallback, interrupt integration with retrieve_called == [] assertion); 66 total passing
- Note: `_compose_answer` does not wire Langfuse callbacks into the Gemini call (minor observability gap, consistent with `_fact_extract` pattern — can add in OBS pass)

### AGENT-4 — Reasoner Rewrite (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_model_claims` removed; `_structured_claims(facts, issue_queries, ranked_hits)` added — Azure `gpt-4.1-mini` via `AzureChatOpenAI`, tier-labelled context (32k char cap), structured output with `issue`/`applicability`/`condition`; `_CONTEXT_CHAR_LIMIT` + `_TIER_LABELS` constants added; `azure_base_url` imported; co-retrieved chunks inherit `_issue_idx` from parent hit
- `query_graph.py`: `retrieve_generate_node` split → `retrieve_node` (pure retrieval, `_issue_idx` tagging) + `reasoner_node` (per-issue `_structured_claims` call over authority-ranked context, grouping by `_issue_idx`); `validate_node` updated to propagate `issue`/`applicability`/`condition` from original claims to rendered results; graph now 7 nodes
- `tests/test_orchestrator.py`: tests 1–3 and 5 updated to mock `_structured_claims`; 2 new tests for `_structured_claims` success path and no-key fallback; 64 total passing
- `tests/test_degraded_modes.py`: stale `_model_claims` mock updated to `_structured_claims` (Pi found this proactively)
- PS-6, PS-7, PS-12 verified; zero-tolerance gates at 0

### AGENT-3 — Authority Ranker + Cross-Reference Resolver (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_authority_rank_hits(hits, conn)` — queries `chunks.work_id → work.work_type` (LEFT JOIN), sorts by `(tier ASC, score DESC)`, attaches `tier` + `conflict_flag` (same section_number, lower tier); `_resolve_cross_refs(hits, as_of, conn)` — regex scans top-10 hits for `दफा/उपदफा/अनुसूची X`, fetches eligible co-chunks via `eligible_chunk_ids`, appends with `co_retrieved=True`; both have `except Exception` top-level guard; `_REAL_MONOTONIC` removed; `_WORK_TYPE_TIER`, `_DEVA_DIGIT_MAP`, `_CROSS_REF_RE` constants added; `eligible_chunk_ids` re-exported for mock compatibility
- `query_graph.py`: `authority_ranker_node` and `cross_ref_resolver_node` inserted between `retrieve_generate` and `validate`; graph now 6 nodes
- `tests/test_orchestrator.py`: 5 new tests; 62 total passing
- Schema note: ADR says `documents.work_type` but correct path is `chunks.work_id → work.work_type`; conflict detection is document-agnostic (same section_number across different works can trigger it — acceptable for Stage 3; Stage 4/5 can scope by work if needed)

### AGENT-2 — Fact Extractor + issue-driven retrieval (MERGED to dev, 2026-08-25)
- `gated_orchestrator.py`: `_fact_extract()` added — Gemini 2.5 Flash extracts `facts`, `missing_facts`, and `issue_queries` (Devanagari Nepali queries, max 3); fallback to single raw query on any failure or missing `GEMINI_API_KEY`
- `query_graph.py`: `classify_node` replaced by `fact_extractor_node`; `retrieve_generate_node` iterates `issue_queries` instead of `subqueries`; `_graph_clock` removed (was CPython-specific frame-walking); `run_query` now uses `_orch.time.monotonic()` for mock-compatible wall_clock_start
- `tests/test_orchestrator.py`: tests 1–3 and 5 updated to mock `_fact_extract`; test 4 unchanged; 2 new tests for `_fact_extract` success path and fallback; 57 total passing
- Cleanup note: `_REAL_MONOTONIC` in `gated_orchestrator.py` (line 19) is now unused — remove in Stage 3 sweep

### AGENT-1 — LangGraph skeleton (MERGED to dev, 2026-08-24)
- `query_state.py`: `QueryState` TypedDict — full schema incl. Stage 2+ placeholders
- `query_graph.py`: 4-node linear graph (classify → retrieve_generate → validate → assemble); `conn` via `config["configurable"]`; all calls via `_orch.*` for monkeypatch compatibility
- `gated_orchestrator.py`: `answer()` delegates to `run_query()`; all helpers remain at module level; `retrieve_postgres` + `validate_and_render` re-exported; `_emit_answer_trace_from_state` extracted
- `requirements.txt`: `langgraph>=1.2`
- 55 tests passing, no behaviour change
- Cleanup note: `_graph_clock` in `query_graph.py` uses `sys._getframe` (CPython-specific, solves non-existent problem in LangGraph 1.2 sync path) — remove in next cleanup cycle

### RET-C — FlashRank fallback reranker (MERGED to dev, 2026-08-24)
- `reranker.py`: full rewrite — Cohere → FlashRank (`ms-marco-MultiBERT-L-12`, multilingual) → passthrough ladder; module-level `_ranker` cache; `except Exception: pass` on Cohere falls through silently
- `requirements.txt`: `flashrank` added
- `tests/test_retrieval.py`: old `test_reranker_skipped_when_cohere_key_unset` replaced with 4 tests covering full ladder; 53 total passing
- No PS requirements in scope; no gates affected (post-retrieval path)

---

## Completed tasks

### RET-B — Dual-path cross-lingual query translation (MERGED to dev, 2026-08-24)
- `postgres_retriever.py`: `_is_devanagari()` (U+0900–U+097F, 0.5 threshold); `translate_query()` via Gemini 2.5 Flash (`langchain-google-genai`); dual-path vector + lexical search when translation succeeds; 4-list RRF fusion; `translation_ran` in eligibility_gate span
- `config.py`: `GEMINI_API_KEY: str = ""`
- `requirements.txt`: `langchain-google-genai>=2.0`
- `tests/test_retrieval.py`: 8 new tests (50 total passing); Cursor mock upgraded to SQL-content detection for dual-path correctness
- Graceful degradation: translation failure → single-path fallback, no exception
- PS-8 served (Romanized Nepali eval slice); no invariants weakened; lint clean

---

## Completed tasks

### OBS-RET — Retrieval observability + eval slice (MERGED to dev, 2026-08-23)
- `postgres_retriever.py`: 6 Langfuse stage spans (eligibility → vector → lexical → RRF → relevance gate → rerank); each with latency_ms, counts, scores
- `gated_orchestrator.py`: answer trace expanded — retrieval/generation/validation latency, claims_passed/abstained, top_chunk_scores
- `romanized_slice.py`: fixed URI matching (source_id based, not URI prefix)
- `retrieval_slice.py`: new — Recall@1/3/5 + MRR; baseline 0.4 / MRR 0.33 on 200 laws
- `eligibility_gate.py`: dropped `valid_time` transaction-time check (was blocking all retrospective queries)
- PS-14 compliant; 42 tests passing

### RET-A — Retrieval layer rewrite (MERGED to dev, 2026-08-23)
- `postgres_retriever.py`: full rewrite — preprocessing, Azure query embedding,
  eligibility gate, vector ANN + tsvector GIN, RRF fusion, relevance gate, Cohere rerank
- `eligibility_gate.py`: new `eligible_chunk_ids()` querying `documents`/`chunks` (not old `lifecycle_effect`)
- `reranker.py`: new Cohere wrapper, opt-in (no-op if `COHERE_API_KEY` unset)
- `validation_gate.py`: resolves evidence_ids against `chunks` table (not `expression`)
- `migrations/007_retrieval_indexes.sql`: GIN tsvector index on `chunks.chunk_text`
- `config.py`: added `COHERE_API_KEY: str = ""`
- PS-6, PS-7, PS-12 verified; 42 tests passing, 2 skipped

### CLEANUP-A — Remove OpenSearch (MERGED to dev, 2026-08-22)
- Deleted: `app/search/client.py`, `app/search/__init__.py`, `app/retrieval/dumb_retriever.py`, `docker-compose.yml`
- `gated_orchestrator.py`: removed `_try_retrieve`, direct `retrieve_postgres` call
- `requirements.txt`: removed `opensearch-py==2.7.1`
- `Makefile`: removed OpenSearch startup from `setup` target
- `tests/test_degraded_modes.py`: deleted 2 OS tests, fixed 1 mock
- Eval slices + `scripts/query.py`: swapped to `retrieve_postgres` + `connect()`
- 33 passed, 2 skipped, 0 failed (count drop = 2 deleted OS tests that were passing)

### PH-OBS-B — Full stage-level ingestion tracing (MERGED to dev, 2026-08-22)
- Replaced terminal-only `_emit_ingestion_span` with per-document traces
- Module-level Langfuse singleton — one client for entire ingestion run
- One trace per document (`ingestion.law` / `ingestion.nkp_case`)
- One timed child span per stage: LOAD, VALIDATE, CHUNK, EXTRACT_METADATA,
  EMBED_AND_UPSERT, DUAL_APPROVAL_PAUSE (+ REDACT_PII for NKP)
- Each span carries: stage name, outcome, latency_ms
- `ImportError` guard — degrades to no-op if langfuse package not installed
- PS-14 compliant; 35 tests passing

### PH-OBS-A — Langfuse RAG tracing integration (MERGED to dev, 2026-08-22)
- `langfuse>=2.0` added to `requirements.txt`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `config.py`
- `LangfuseCallbackHandler` wired into LangChain LLM calls in `gated_orchestrator.py`
- `answer()` emits trace: `query_hash` (SHA-256 only), `as_of`, `query_type`,
  `latency_ms` (full elapsed), `retrieved_uris` (component_uri only), `gate_decision`, `result_count`
- `IngestionPipeline` emits per-stage spans: `source_id`, `source_type`, `stage`, `outcome` — no raw content
- PS-14 compliant: no raw query text or statutory text in any span
- Opt-in: no-op when `LANGFUSE_PUBLIC_KEY` unset
- 35 tests passing, lint clean

### PG-B — Azure OpenAI + laws ingestion (2026-08-07, on dev)
- Switched embeddings: `BAAI/bge-m3` (local SentenceTransformer) → Azure OpenAI `text-embedding-3-large`
  - `dimensions=1024` preserves existing schema; no migration required
  - `DEFAULT_BATCH_SIZE` raised from 32 → 512 (no GPU memory constraint with API)
- Switched LLM: standard OpenAI → Azure OpenAI `gpt-4.1-mini`
  - `AzureChatOpenAI` in `metadata_enricher.py`; `AzureOpenAI/AsyncAzureOpenAI` in `ragas_eval.py`
  - Separate `AZURE_OPENAI_LLM_KEY` + `AZURE_OPENAI_LLM_ENDPOINT` fields (distinct from embedding)
  - `azure_base_url()` helper in `config.py` strips deployment path from full endpoint URL
- `sentence-transformers` removed from `requirements.txt`
- Ingestion optimisations:
  - Chunk-metadata LLM call batched at 20 chunks/call (was unbounded — caused 2+ min hangs on large acts)
  - Batches parallelised via `ThreadPoolExecutor(max_workers=3)` — ~2.5× speedup on large acts
  - 60s timeout on `AzureChatOpenAI` (was SDK default 600s — caused silent 10-min hangs)
  - OOM retry loop removed from `_embed()` — irrelevant for API calls
- `scripts/ingest_laws.py`: per-record progress printed to stdout (`[N/total] source_id … ✓ ingested`)
- **100 laws ingested** to local Postgres: 4,263 chunks, Nepali summaries + keywords generated
- Est. cost for 100 laws: ~$1.10 (LLM ~$0.90 + embeddings ~$0.20)

### PG-A — RAGAS v0.2 eval slices per pipeline phase (MERGED to dev, 2026-08-06)
- `ragas==0.2.*` added to `requirements.txt`
- `app/eval/ragas_eval.py`: `BaseRagasLLM` + `BaseRagasEmbeddings` via `openai.AsyncOpenAI`
  directly — no LangchainLLMWrapper, no langchain-openai version conflict
- `app/eval/__init__.py`: minimal shim for `langchain_community.chat_models.vertexai`
  (removed in langchain-community 0.4.x; stub lets ragas 0.2.* import cleanly)
- Phase slices: phase_a (Faithfulness + ResponseRelevancy), phase_c (ContextRecall +
  NonLLMContextPrecisionWithReference), phase_d (stubbed — skips if precedent empty),
  phase_ef (summary Faithfulness vs. source chunks)
- `app/eval/metrics/temporal_faithfulness.py`: custom PS-6-aligned LLM-judge metric
- Golden sets: `phase_a_qa.json` (10), `phase_c_romanized.json` (10), `phase_d_precedent.json` (5 placeholders)
- `make eval`: runs all slices + romanized Recall@5; `make eval-gates` unchanged
- PS-6, PS-13 in scope; zero-tolerance gates all at 0

### P0-A — Infrastructure skeleton (MERGED, commit 553a4e3)
- Makefile, bitemporal schema, OpenSearch client, Pydantic models

### P0-B — Full corpus ingestion + dumb baseline (MERGED, commit 9f0f086 + beeb05bf + 9b08516)
- `app/authority/parser.py`, `writer.py`, `eligibility_gate.py`, `dumb_retriever.py`,
  `validation_gate.py`, `eval/gates.py`, `scripts/ingest_laws.py`, `scripts/query.py`
- migration 002: eligibility gate suspend fix

### PA-A — Wire /ask with bitemporal gated pipeline (MERGED, commit d270be2)
- `app/main.py`: eligibility gate → dumb_retriever → model (claims+evidence_ids) → validate_and_render
- `tests/test_ask_pipeline.py`

### PB-A — Gated orchestrator + degraded-mode ladder (MERGED, commit 5dfbfcc)
- `app/retrieval/gated_orchestrator.py`, `app/retrieval/postgres_retriever.py`
- Degraded modes: Postgres-down→503, OpenSearch-down→Postgres fallback, model-down→extractive
- 10 new tests

### PC-A — Canonical BS/AD calendar + romanized eval slice (MERGED, commit 4b74709)
- `app/authority/bs_ad_calendar.py`: BS 2000-2090 lookup, BeyondCalendarRange, boundary-window infra
- `app/authority/parser.py`: bs_to_ad_approx() deleted; canonical lookup in place
- `migrations/003_bs_ad_calendar.sql`, `scripts/seed_bs_ad_calendar.py`
- `app/eval/romanized_slice.py`: Recall@5 harness; 10 golden queries
- 9 new tests (24 total passing)

### PD-A — Precedent schema + eval gate + retriever skeleton (MERGED, commit 96533eb)
- `migrations/004_precedent_schema.sql`: precedent/holding/relation tables + `is_good_law(uuid, date)`
- `app/authority/precedent_models.py`, `app/retrieval/precedent_retriever.py`
- `app/eval/gates.py`: `check_overruled_as_good_law()` wired
- 5 new tests (29 total passing, 1 skipped)

### PE-A — PostgreSQL + pgvector + bge-m3 ingestion pipeline (MERGED to dev, 2026-08-03)
- `migrations/005_ingestion_pipeline.sql`: documents, chunks (pgvector 1024-dim, HNSW), pii_vault,
  pg_search BM25 index; dual-approval DDL (PS-2); REVOKE ALL on pii_vault (PS-14)
- `app/ingestion/laws_chunker.py`: दफा-anchor structure-aware chunker; PS-16 co-retrieval links
- `app/ingestion/nkp_chunker.py`: hybrid anchor chunker (caption/headnote/opinion/order/colophon)
- `app/ingestion/pii_redactor.py`: deterministic + LLM second pass + verification assertion
- `app/ingestion/pgvector_indexer.py`: bge-m3 embed, chunk upsert in index order, pii_vault write
- `app/ingestion/pipeline.py`: 8-stage orchestrator; never sets approved; quarantines on redaction failure
- `scripts/ingest_laws.py`, `scripts/ingest_nkp.py`: CLI scripts with dry-run mode
- `tests/test_ingestion_pipeline.py`: 35 passing, 2 skipped
- PS-2 / PS-3 / PS-5 / PS-10 / PS-14 / PS-16 all verified GREEN

### PE-A/fix — Provider-agnostic LLM via LangChain 1.3.0 (MERGED to dev, 2026-08-03)
- `metadata_enricher.py`, `pii_redactor.py`: `anthropic` SDK replaced with `init_chat_model(settings.LLM_MODEL)`
- `config.py`: `LLM_MODEL: str = "openai:gpt-4o-mini"` — swap provider via env var, no code change
- `requirements.txt`: langchain==1.3.14, langchain-openai==1.4.1, langchain-community==0.4.2; anthropic removed
- Collateral: `langchain.schema.Document` → `langchain_core.documents.Document` (removed in LangChain 1.x)

### PF-A — Local Postgres setup + summary field (2026-08-06, on dev)
- **Local Postgres**: Docker container `wakilg-postgres` (pgvector/pgvector:pg17, port 5433)
  - All 13 tables created; pgvector extension live; 33,238 BS/AD calendar rows seeded
  - `DATABASE_URL=postgresql://wakilg:wakilg@localhost:5433/wakilg` in `.env`
- **`app/authority/writer.py`**: reads `DATABASE_URL` first, falls back to `SUPABASE_DB_URL`
- **`migrations/006_add_summary.sql`**: `ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT`
- **`app/ingestion/metadata_enricher.py`**: summary added to both enrichment paths
  - NKP: extracted in same first LLM call as `cited_statutes` + `headnotes` (no extra API call)
  - Laws: one extra LLM call per act (act name + first 5 chunks → 2-3 sentence Nepali summary)
- **`app/ingestion/pipeline.py`**: extracts `summary` from enricher output, passes to document dict
- **`app/ingestion/pgvector_indexer.py`**: writes `summary` into documents upsert
- **`scripts/migrate.py`**: fully rewritten — idempotent via `schema_migrations` tracking table;
  detects pre-existing migrations by object probes; BM25 block auto-skipped on standard Postgres
- Dry-run verified: 1022 NKP cases in `output/nkp_cases.jsonl`, 5/5 sample valid, 0 rejected

## Phase 0 + A + B + C + D + E + F status
**Schema and pipeline COMPLETE. Corpus ready. Ingestion not yet run.**
- All gates enforced on every path (including all degraded modes)
- Canonical BS/AD calendar live (PS-5); romanized eval slice live (PS-8)
- Precedent subsystem with holding-level model and bench-competence gate (PS-1)
- Ingestion pipeline: PostgreSQL + pgvector + bge-m3; dual approval enforced in DDL
- Documents carry: keywords, relevant_questions, cited_statutes, headnotes, **summary**
- Zero-tolerance gates: repealed-as-current = 0, not-yet-effective-as-current = 0, overruled-as-good-law = 0

## Operational steps still pending (on Prakash)
- Run `python3 scripts/cleanup_stale_authority_expressions.py --dry-run`
  then for real, against the live local DB (AGENT-17) — reconciles the
  authority store to the fixed दफा/अनुसूची parser; local Postgres wasn't
  running in either review session to execute this
- Run `scripts/ingest_laws.py` for remaining 577 laws (100 done, 677 total)
- Run `scripts/ingest_nkp.py --input output/nkp_cases.jsonl` (1022 NKP cases)
- Run `make eval` + `make eval-gates` against live env (baseline Recall@5 + zero-tolerance gate check)
- Ingest precedent corpus (then wire `retrieve_precedent` into orchestrator)
- Rewrite `app/main.py` auth layer (Supabase auth → new architecture; `app/utils/helpers.py` SupabaseHelper to be replaced)
- Push `dev` to origin when ready

## Architecture notes
- **DB**: Self-hosted PostgreSQL on VPS (Docker locally). No Supabase dependency for ingestion or retrieval.
  `app/main.py` still has Supabase auth — that is old architecture, to be replaced.
- **LLM**: Azure OpenAI `gpt-4.1-mini` via `AzureChatOpenAI`. Keys: `AZURE_OPENAI_LLM_KEY` + `AZURE_OPENAI_LLM_ENDPOINT`.
- **Embeddings**: Azure OpenAI `text-embedding-3-large` at `dimensions=1024`. Keys: `AZURE_OPENAI_KEY` + `AZURE_OPENAI_ENDPOINT`.
- **BM25**: pg_search (ParadeDB) not available on standard Postgres — falls back to GIN tsvector at query time.

## Governing design refs
- AGENTS.md (prime directive, definition of done)
- docs/ingestion_design.md (PE-A design; approved by Prakash 2026-08-02)

## Next action
Run Pi on `agent/authority-linked-citations` (AGENT-22, re-dispatched
2026-09-01) — branch cut fresh off `dev`, same scope, task.md carries an
added note asking the engineer to verify `git status` on this exact
branch before reporting completion.
`_fetch_enabling_chunk` (`query_graph.py`) bypassing the eligibility gate
for co-retrieved enabling provisions is still open, still not a task.
Operational, still pending: `documents`/`lifecycle_effect` rows in the
live local DB have never had a document-level approval run against them
— run `scripts/review_documents.py --list` against it. Backfill scripts
from AGENT-22 (once redone) and AGENT-23 (`backfill_source_kind.py`,
`recompute_content_hashes.py`, both ready now) still need a live DB —
local Postgres wasn't running in this session either (`localhost:5433
connection refused`), same operational gap noted for AGENT-17's cleanup
script.
