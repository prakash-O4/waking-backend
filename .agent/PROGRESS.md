# Wakil-G — Orchestration Progress

## Current task
None.

## Status
**IDLE** — AGENT-16 merged to dev. Awaiting Prakash's direction.

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
- Planned follow-on tasks (not yet branched):
  - **AGENT-17** — strengthen VALIDATE stage (structural checks beyond दफा
    anchor: duplicate/broken section numbering, malformed markup, doc-type
    support, chunk-size violations) + named tariff-threshold constant
    (currently a bare `5000` literal in `app/ingestion/tariff_chunker
    .py:38`). Split out of the originally-planned AGENT-14 scope
    (2026-08-30) — the concrete, evidenced part (header regex
    over-matching + hash canonicalization) became AGENT-14 on its own;
    this is the remaining open-ended hardening work.
  - **AGENT-18** — clause-level `<amend>` tag extraction: correlate each
    inline `<amend>...</amend>` marker with the enclosing दफा and with
    the document's own "संशोधन गर्ने ऐन"/"संशोधन" amendment-history
    table (by act name for the 83% dominant pattern, by Devanagari
    ordinal-word position — "पहिलो", "दोस्रो", "आठौं" — for a real
    secondary pattern where the tag references the table entry by
    position instead of by name) to produce `amend`-type lifecycle
    facts. Renumbered out of the old "AGENT-16" label during AGENT-16's
    scoping (2026-08-31) — see that entry above for the corpus
    grounding (15,748 tag instances / 519 documents / dominant-pattern
    breakdown) and the 1% gazette-notification-only sub-pattern
    (no named act, just a date, mirrors AGENT-12's gazette-dependent
    commencement shape).

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
Awaiting Prakash's direction.
