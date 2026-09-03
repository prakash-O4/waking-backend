# Wakil-G — Orchestration Progress

## Current task
None dispatched. AGENT-30 (task 5 of 6 in the retrieval-hardening program)
is **MERGED**. AGENT-31 (real stress/red-team suite) is next and last —
scoping/task.md not yet written, queued.

## Status
**IDLE, between waves** — AGENT-30 merged to `dev`. AGENT-31 is the final
task in the retrieval-hardening program. Awaiting Prakash's go-ahead to
scope and dispatch it.

## Retrieval-hardening program (started 2026-09-02, target: 4/10 → 9/10)

Grounded against an external review, verified claim-by-claim against actual
code before accepting any of it (full grounding + review-claim-by-claim
verification in the dated entry below). Sequenced by file overlap — same
file means same engineer, in order, not parallel (per the AGENT-19/20
lesson: two engineers colliding on one file). Three design questions were
already asked and answered by Prakash before this table was cut — do not
re-ask them:
- `suspend` **does** terminate eligibility during its window (matches
  migration 002 + `validation_gate.py`'s prior behavior).
- Claim-support check is a **deterministic verbatim-quote substring match**,
  not an NLI/semantic-entailment model — no new dependency.
- Phase D (precedent) is **not** active — leave `retrieve_precedent()`
  unwired into `/ask` until Prakash explicitly says otherwise.

Also explicitly out of scope, not tasks: jurisdiction filtering
(`work.jurisdiction` is `'NP'` on every row today — no-op until
multi-jurisdiction is real) and ACL (no schema concept exists — public
legal-QA product, no per-user document permissions).

| # | Status | Scope | Files | PS / Invariant | Depends on |
|---|---|---|---|---|---|
| AGENT-26 | **MERGED** (`66ad5e9`) | Canonical eligibility predicate — wire `eligible_chunk_ids()` + `_terminated_before()` to the DB's `is_eligible()`, single source, no drift | `eligibility_gate.py`, `validation_gate.py::_terminated_before` | CI #1,#2; PS-2,PS-4,PS-15 | none — foundational |
| AGENT-27 | **MERGED** (`c1304cf`) | Claim-support: model emits a verbatim quote alongside each claim; server substring-checks it against `chunk_text` | `validation_gate.py`, `gated_orchestrator.py::_structured_claims` (prompt), `query_graph.py` (claim shape) | CI #4,#7 | AGENT-26 (same file — sequenced) |
| AGENT-28 | **MERGED** (`c6923ba`) | Citation authority-chain: resolve `component`→`expression`→amending `lifecycle_effect`, label `derived`, stop reading raw chunk metadata as the citation | `validation_gate.py::_citation` | PS-3 | AGENT-26/27 (same file — sequenced) |
| AGENT-29 | **MERGED** (`4092fcb`) | Composer output re-validation: strip/abstain any composed section whose citation doesn't match a validated `evidence_id` | `gated_orchestrator.py::_compose_answer`, `query_graph.py::answer_composer_node` | CI #3,#4 | none — different file, parallel-safe |
| AGENT-30 | **MERGED** (`f1ed7cb`) | Exact दफा/धारा/उपदफा/अनुसूची/Act-title lookup merged into `retrieve_postgres` alongside vector+lexical via the existing `_rrf` | `postgres_retriever.py` | (retrieval precision) | none — different file, parallel-safe |
| AGENT-31 | queued, run last | Real stress/red-team suite: repealed/current, not-yet-effective, romanized, cross-ref, proviso, enabling-power taxonomy cells | `Makefile`, new `tests/stress/` | PS-12 | should land after 26-28 so it tests the *fixed* invariants, not the current gaps |

**Next action**: scope AGENT-31 — the final task in the program (real
stress/red-team suite: repealed/current, not-yet-effective, romanized,
cross-ref, proviso, enabling-power taxonomy cells — write `task.md`, create
`agent/stress-redteam-suite` off `dev`), dispatch to Pi.

- **AGENT-30 review round 2 (2026-09-03, MERGED)**: Pi returned `973e1f6` —
  real commit, correct branch, clean tree. Diffed `a2354cb..973e1f6`
  specifically (not `bf17e64..973e1f6`, which would have included my own
  already-reviewed rework-note commit) to see just Pi's actual fix: a
  clean two-line split — `_parse_section_reference` now matches only
  `(?:दफा|धारा)\s*(\d+)`, and a new, separate `_parse_subsection_reference`
  handles उपदफा on its own, never feeding `section_num`. Exactly the fix
  requested, nothing more. Reproduced both rework-note repro cases myself
  directly against the fixed function rather than trusting the diff read:
  `_parse_section_reference("उपदफा (2) मा के छ?")` → `None` (was `"2"`),
  `_parse_section_reference("उपदफा (2), दफा 9 अनुसार")` → `"9"` (was
  `"2"`) — both now correct. All three required tests present and
  behavioral: the fixed `test_parse_section_reference` assertions, a new
  direct `test_parse_subsection_reference`, and a new
  `test_bare_subsection_does_not_run_exact_section_filter` proving
  end-to-end that a bare उपदफा query returns `[]` and never even reaches
  the exact-lookup SQL (`not any("num" in params for params in
  params_history)`) — the exact "doesn't silently filter by the wrong
  thing" proof the rework note asked for, not just a unit test on the
  regex in isolation.
  One thing considered and explicitly not sent back: `_parse_subsection_reference`
  is defined and tested but never called from `retrieve_postgres()` itself
  — not wired into any variable or observability span. My rework note's
  wording ("parse it... needed so the completion report can note the
  schema limitation") was genuinely ambiguous about whether it needed to
  be wired into live code or just exist as a documented, tested capability;
  read it as the latter — the substance of the finding was the *filter*
  conflation, which is fully fixed and tested, and this doesn't affect
  correctness, so treating it as a second finding would be nitpicking
  beyond what actually matters.
  Independently re-verified rather than trusting the report: `make test`
  (218 passed, 3 skipped — matches), `make lint` clean, `make eval-gates`
  0/0/0 (live DB, all reproduced myself). No further findings. Merged
  `agent/exact-citation-lookup` → `dev` (`--no-ff`, `f1ed7cb`).

- **AGENT-30 review round 1 (2026-09-03, rework requested)**: Pi returned
  `bf17e64` — real commit, correct branch, clean tree, file scope exactly
  matched `task.md` (`postgres_retriever.py` + `tests/test_retrieval.py`
  only). Independently re-verified rather than trusting the report:
  `make test` (216 passed, 3 skipped — matches), `make lint` clean, `make
  eval-gates` 0/0/0 (live DB, all reproduced myself). Most of the diff is
  genuinely correct: `exact_lookup_search()`'s SQL matches `task.md`'s
  8-column shape exactly (no `_hit()` changes needed); the `rows`/
  `ranked_lists` KeyError trap the brief specifically warned about was
  correctly avoided (`exact_rows` folded into both, with a dedicated test
  — `test_exact_lookup_only_result_survives` — proving an exact-only hit
  with empty vector/lexical arms survives to the final output); `strpos()`
  used correctly for Act-title matching, longest-match-wins proven by a
  real test with two overlapping titles; Pi independently made a correct
  judgment call I hadn't fully specified — Act-title resolution runs on an
  NFC-only (not digit-folded) copy of the query, since `work.title_ne`
  values in the DB retain their original Devanagari-digit years and a
  digit-folded query would fail to substring-match them; eligibility gate
  correctly applied to the new query branch (Core Invariant #2 intact);
  the same RRF/relevance-threshold/rerank pipeline is used, no bypass lane
  added.
  **One real finding, sent back rather than merged**: `_parse_section_reference`
  matches `(?:दफा|धारा|उपदफा)\s*\(?(\d+)\)?` — all three words share one
  capture group, so a bare उपदफा's own number can be returned as
  `section_num` and fed into the दफा/धारा filter as if it were a दफा
  number. उपदफा numbers are enumerated per-दफा (दफा 5 and दफा 9 each have
  their own उपदफा (1),(2),(3)...) — reusing one as the other is not an
  approximation, it's a category error, exactly what `task.md`'s
  schema-grounding section had explicitly warned against before this was
  dispatched. Reproduced live myself before writing the finding, not just
  reasoning about the regex: `_parse_section_reference("उपदफा (2) मा के
  छ?")` → `"2"` (should be `None` — no दफा/धारा number was ever stated);
  `_parse_section_reference("उपदफा (2), दफा 9 अनुसार")` → `"2"` (should be
  `"9"` — `re.search` takes the leftmost match, and उपदफा appearing first
  in the sentence silently wins over the actual दफा number). This currently
  ships **locked in by a passing test**
  (`test_parse_section_reference`'s `_parse_section_reference("उपदफा (2)")
  == "2"` assertion actively asserts the wrong behavior) — not an
  oversight, a confirmed design choice that deviates from the brief.
  Rework note appended to `task.md` (`a2354cb`) — required fix: drop
  उपदफा from `_parse_section_reference`'s regex entirely (match only
  `(?:दफा|धारा)\s*(\d+)`), parse उपदफा separately as a non-filtering
  signal per the original brief's letter, fix the now-wrong test
  assertion, and add tests proving both the word-order case and the
  bare-उपदफा-only case resolve `section_num` correctly (`"9"` and `None`
  respectively, not `"2"`). Same branch, same engineer, per the Rework
  Loop — not re-scoped.

- **AGENT-30 scoping (2026-09-03)**: grounded in the actual chunk schema
  before writing the brief, not just the design doc's one-liner. Traced
  `laws_chunker.py`'s `LawChunk.parent_section` docstring ("दफा number when
  level='subsection'/'proviso'") to confirm `chunks.section_number` +
  `chunks.parent_section` together cover दफा/धारा exact matching (both
  words store their number the same way — the distinction is which
  document uses which term, not a schema field) — but there is **no
  separate उपदफा-number column**, a real, honest schema limitation stated
  plainly in `task.md` rather than papered over: उपदफा is parsed from the
  query for completeness but doesn't add filter precision beyond the दफा
  number it's nested under. Checked `tariff_chunker.py` before scoping
  अनुसूची matching — confirmed ordinary Acts have **no structured schedule
  column at all**; only tariff-schedule Acts model schedules, via a
  completely separate `TariffChunk` system this task must not touch.
  Scoped अनुसूची matching down to what the schema can actually back: a
  literal `strpos()` containment check for "अनुसूची <N>" against
  `chunk_text`, explicitly flagged in `task.md` as lower-precision than the
  दफा/धारा path, not pretended otherwise. Confirmed `chunks.work_id`
  already exists directly on the chunk row (no need to route Act-title
  filtering through `component`/`component_uri`). Specified `strpos()`
  over `ILIKE`-with-concatenated-wildcards for the Act-title lookup
  specifically to avoid LIKE-pattern-injection risk from title text
  flowing into a pattern position — a real, if low-probability, class of
  bug, sidestepped entirely rather than patched with `ESCAPE`.
  Read `tests/test_retrieval.py` before writing the allowed-scope/test
  section — its `Cursor` mock dispatches on SQL substrings (`"embedding
  <=>"` vs `"ts_rank_cd"`); pointed Pi at extending that same dispatch
  pattern with a third branch rather than building a parallel mock, and
  explicitly required proving the existing vector/lexical dispatch still
  works unchanged.
  Central design constraint carried over from the review's own program
  table (not reinvented here): exact matches join the **same** `_rrf()`
  fusion as vector/lexical, no bypass lane, no skipped relevance threshold,
  no skipped rerank — verified the math holds on its own (an exact match
  ranked #1 in its own one-item list already scores `1/(60+1) ≈ 0.0164`,
  comfortably above the existing `0.005` relevance threshold) so no
  special-casing is needed for a lone exact match to survive the existing
  gate.
  One correctness trap flagged explicitly in `task.md`: `rows` (the dict
  `_hit()` reads from) is currently built only from `vector_rows`/
  `lexical_rows`/their translated variants
  (`postgres_retriever.py:238-241`) — a chunk id found only via exact
  lookup and folded into `ranked_lists` without also being added to `rows`
  would `KeyError` at the candidate-building step. Required a dedicated
  test seeding exactly this case (exact-only hit, empty vector/lexical
  arms) survives to the final output.
  Branch `agent/exact-citation-lookup` created off `dev`. `task.md`
  committed there (`d3426e8`, author `Prakash Basnet`). Assigned to Pi.
  Awaiting Prakash to dispatch.

- **AGENT-29 (2026-09-03, MERGED)**: Pi returned `bb6c7d2` — real commit,
  correct branch, clean tree. File scope matched `task.md` exactly
  (`gated_orchestrator.py`, `query_graph.py`, `tests/test_orchestrator.py`
  — the existing test file, not a new one, per the brief's pointer).
  Prompt correctly rewritten to ask for `evidence_id`/`as_of` instead of a
  model-authored `"citation": {}`. New `_revalidate_composed()` builds its
  lookup keyed by `(evidence_id, as_of)` from `all_results`, filtering to
  non-abstained entries only — verified the compound-key requirement is
  real, not just present in a comment, via the diff and a dedicated test
  that seeds the *same* `evidence_id` under two different `as_of` values
  with genuinely different citations and proves the correct one is picked
  by identity (`is new`, not just `==`). Every matched section's
  `citation` is unconditionally overwritten with the canonical dict —
  confirmed the model's own citation content is never trusted even when
  present, via a test that seeds a `{"source": "model"}` citation on a
  matching section and asserts the final result `is` the distinct
  server-side object. `abstained` is always recomputed from the
  post-filter section count (never trusted from the composer's JSON) and
  `plain_language` blanks only on total strip, left alone on partial
  strip — both proven with dedicated tests, matching the design decisions
  in `task.md` exactly. Malformed input handled defensively and tested: a
  non-list `relevant_sections`, and a non-dict item inside a list, are
  both treated as empty/skipped rather than raising. The adjacent
  one-line fallback-abstention bug (`query_graph.py:375`, `not
  all_results` → `not any(not r.get("abstained") for r in all_results)`)
  is fixed exactly as specified and covered by its own test (non-empty
  all-abstained list correctly reports `abstained: True` now). A
  full node-level test (`test_answer_composer_node_revalidates_composed_output`)
  proves the wiring itself, not just the helper function in isolation —
  drives `answer_composer_node` end-to-end with a monkeypatched
  `_compose_answer` returning a bogus model citation and confirms the
  final `_response`'s citation is the canonical server object.
  Independently re-verified rather than trusting the report: `make test`
  (211 passed, 3 skipped — matches), `make lint` clean, `make eval-gates`
  against the live DB (back up this session after being unreachable during
  AGENT-28's review) — `repealed-as-current: 0`,
  `not-yet-effective-as-current: 0`, `overruled-as-good-law: 0`, all
  matching. No findings — nothing sent back. Merged
  `agent/composer-output-revalidation` → `dev` (`--no-ff`, `4092fcb`).

- **AGENT-29 scoping (2026-09-03)**: traced `answer_composer_node`
  (`query_graph.py:381`) end-to-end — its returned `_response` **is** the
  literal `/ask` API response (`build_graph()`'s final
  `return cast(dict[str, Any], result["_response"])`). Confirmed the live
  gap directly: `_compose_answer`'s prompt (`gated_orchestrator.py:213-231`)
  asks a second, independent Gemini call to write its own
  `"citation": {}` per `relevant_sections` entry, restrained only by
  prompt text ("Never modify citations") — no server-side check exists
  anywhere between that call returning and it becoming the response.
  Design choice made directly rather than left to Pi: join key for
  matching a composed section back to its source claim is
  **`(evidence_id, as_of)`**, not `evidence_id` alone — a diachronic query
  can carry the same `evidence_id` under two different per-claim `as_of`
  values (Core Invariant #6) with genuinely different citation content, so
  a single-field key could silently attach the wrong `as_of`'s citation.
  Both fields already exist on every `all_results` entry, so this costs
  nothing extra. Also decided: `abstained` is always server-recomputed
  from the post-filter `relevant_sections` list, never trusted from the
  composer's own JSON (same principle as Core Invariant #7, one layer up);
  `plain_language` is blanked only when *all* sections get stripped, left
  alone on partial stripping (flagged as an accepted, not-fixed-here
  limitation — surgically editing prose to remove one section's mention is
  a separate, harder problem).
  Found one adjacent, genuinely-existing one-line bug in the exact same
  function while grounding this: the `composed is None` fallback branch
  (`query_graph.py:375`) sets `"abstained": not all_results` — true only
  when the *list* is empty, not when every entry in a non-empty list is
  individually `abstained: True`. Folded the fix into this task's scope
  (same file, same function, same "abstained must reflect real evidence,
  not list-shape" principle already being applied everywhere else here) —
  not a separate task, too small and too on-theme to warrant one.
  Checked the repo's existing test convention before writing the allowed-
  scope list: `tests/test_orchestrator.py` already covers
  `gated_orchestrator.py` (including `test_compose_answer_success`,
  imports it as `orchestrator`) — pointed Pi at that file, not a new one.
  Confirmed the existing `test_compose_answer_success` only asserts on
  `_compose_answer`'s return shape, not prompt content or citation
  fields — the prompt-schema change this task makes won't break it.
  Branch `agent/composer-output-revalidation` created off `dev`. `task.md`
  committed there (`199420d`, author `Prakash Basnet`). Assigned to Pi.
  Awaiting Prakash to dispatch.

- **AGENT-28 (2026-09-03, MERGED)**: Pi returned `39505c7` — real commit,
  correct branch, clean tree. File scope matched `task.md` exactly
  (`validation_gate.py` + its test file only, no other files touched).
  `_citation()` correctly resolves the TEXT `component.uri` via the
  existing `_authority_component_uri()` helper before touching
  `component`/`work`/`lifecycle_effect` — traced every new query's
  parameter back to its source myself to confirm no conflation between the
  chunk UUID (`evidence_id`) and the real component URI, the specific trap
  the brief called out. Base source query stays keyed by `evidence_id`
  (correct — that's still the right way to find the specific document
  backing this chunk's exact span); `component`+`work`+`lifecycle_effect`
  queries all keyed by the resolved TEXT uri. Linked/unlinked branches
  return identical key sets (proven by a dedicated test, not just visual
  inspection). Amend-chain query correctly restricts to `effect_type =
  'amend'` only, with a comment-free but correct rationale (repeal/expiry/
  suspend/declared_invalid claims never reach `_citation()` at all —
  `validate_and_render`'s `ok` chain already excludes them upstream).
  `derived` flag computed correctly as `source_kind in
  {"verified_internal_consolidation", "derived_verified"}`.
  **Checked a specific concern myself before trusting it**: the new
  amend-chain query compares `lower(le.legal_valid_time) <= %(as_of)s`
  without the explicit `::timestamptz` cast used everywhere else in this
  codebase (`is_eligible()`, `_terminated_before()`). Rather than assume
  either "obviously fine" or "obviously a bug," checked psycopg2's actual
  parameter adaptation directly (`psycopg2.extensions.adapt(date(...))
  .getquoted()` → `b"'2024-01-01'::date"`) — confirmed psycopg2 sends an
  explicitly-typed `date` literal, and PostgreSQL has native
  `timestamptz`↔`date` comparison operators (via `date2timestamptz`
  conversion at the session timezone), so this resolves correctly without
  the cast — a benign style deviation from precedent, not a bug. Wanted to
  additionally verify against the live DB directly (this repo's established
  practice all session) but **could not** — Postgres at
  `localhost:5433` (per `.env`) was unreachable this session (no Docker
  daemon running, no local Postgres process found) — noting this
  explicitly rather than silently skipping it: the amend-chain SQL's
  correctness rests on the psycopg2/Postgres semantics check above, not on
  a live query run against real `lifecycle_effect` rows this session. Also
  confirms `make eval-gates` (which Pi's report cites as green, and which
  did run successfully during AGENT-27's review — DB has since gone
  unreachable in this session) **does not exercise `_citation()` at all**
  (`app/eval/gates.py` only tests `is_eligible()`/`eligible_chunk_ids()`
  against synthetic data, confirmed by grep — same pre-existing gap AGENT-26
  already documented) — so its green result, while real, doesn't itself
  validate this task's new SQL either way.
  **One trivial issue found and fixed directly rather than sent back**: the
  `component`+`work` join selected `c.component_type, c.number` (matching
  this task's own brief, which asked to resolve them) but never used either
  value anywhere in the output — dead columns in the SELECT, my own
  ambiguity in `task.md` (the brief asked to resolve them but never
  specified an output key for them). Trimmed the SELECT to just
  `w.title_ne, w.title_en` (`402f054`) — mechanical, re-ran `make test`
  (206 passed, matches, unaffected) and `make lint` (clean) after, not
  new engineer work worth a round-trip. **Also corrected a `git add -A`
  slip of my own**: my first fixup commit accidentally staged the untracked
  `docs/legal_rag_ingestion_best_practices.md` (the tariff-ingestion notes
  Prakash asked to leave untracked) — caught immediately via `git status`
  before pushing anywhere, `git reset --soft` + `git restore --staged` to
  undo, recommitted with only the intended file. File is back to untracked,
  unaffected.
  Test rewrite is substantively good, not just updated for new fields: new
  `CitationConn`/`CitationCursor` actually evaluates the amend-chain WHERE
  predicate against seeded rows (effect_type/approval_status/as_of-lower
  comparison + sort), not canned booleans — proves, with real seeded data,
  that a future-effective approved amendment, a pending-approval amendment,
  and an approved repeal are each correctly excluded from the same list
  that correctly includes two in-force amendments in effective-date order
  (seeded out of order). Separate tests prove `derived` for a consolidation
  base, linked/unlinked key-set parity, and `None` for an unknown chunk.
  Independently re-verified (after my own fixup): `make test` (206 passed,
  3 skipped — matches), `make lint` clean. `make eval-gates` **not**
  re-run this session (DB unreachable, see above) — flagging this as the
  one check not independently reproduced, rather than claiming it was.
  Merged `agent/citation-authority-chain` → `dev` (`--no-ff`, `c6923ba`).

- **AGENT-28 scoping (2026-09-03)**: grounded against the actual schema
  (`migrations/001_bitemporal_schema.sql`'s `component`/`work`/
  `source_publication`/`lifecycle_effect`/`expression`,
  `migrations/011_chunk_authority_links.sql`'s `chunks.component_uri`/
  `documents.source_pub_id`) rather than the design doc alone. Traced the
  current `_citation()` end-to-end: it joins `chunks`→`documents`→
  `source_publication` by the chunk's own id and reads
  `act_name`/`case_id`/`source_type` off the chunk row directly — never
  touches `component`, `work`, or `lifecycle_effect`, so a claim citing a
  once-amended provision shows only whichever single document backs that
  chunk, with no record of the amendment. Confirmed via
  `grep`/read that `_citation()`'s parameter is misleadingly named
  `component_uri` but is actually called with the chunk's UUID id
  (`claim["evidence_id"]`) at its one call site in `validate_and_render()`
  — the real TEXT `component.uri` only exists via the already-built
  `_authority_component_uri()` helper (added under AGENT-26). This
  conflation is pre-existing and repo-wide (same pattern in
  `eligible_chunk_ids()`'s returned set, retrieval hit dicts, etc.) —
  scoped the fix narrowly to renaming just `_citation()`'s own parameter to
  `evidence_id`, not a repo-wide rename (Ponytail: smallest local change).
  Confirmed nothing downstream pattern-matches specific citation dict keys
  (`grep` for `["citation"]`/`.get("citation")` outside
  `validation_gate.py` — none; the whole `all_results` list is
  JSON-serialized wholesale into the composer prompt) — extending the
  return shape with `derived`/`amendments`/`source_url` is additive, not a
  breaking change to any consumer.
  Confirmed `_citation()` is only ever reached after `validate_and_render`'s
  `ok` chain already passed eligibility + termination — so a
  currently-repealed/terminated component's claim never reaches
  `_citation()` at all, meaning the new amending-chain query only needs
  `effect_type='amend'`, not the full repeal/expiry/suspend/
  declared_invalid list (those are handled upstream already).
  One real, separate finding surfaced while grounding this (not folded into
  AGENT-28, explicitly flagged in `task.md` as forbidden-to-fix-here and
  worth Prakash's attention): `upsert_expression()`
  (`app/authority/writer.py:276`) is only ever called once, during initial
  document `PERSIST_AUTHORITY` ingest (`pipeline.py:306`) — nothing in the
  codebase calls it again when a `lifecycle_effect` amend is later
  approved. Combined with `chunks.chunk_text` also being written once at
  initial ingest, this raises the question of whether retrieved chunk text
  for an amended provision can go stale relative to an approved amendment —
  `system-design.md` §5 says ingestion should "materialize expression;
  re-embed only affected components" on a lifecycle write, but no live code
  path does this today. Not verified against the live corpus this session
  (would need a real amended-and-then-later-ingested component to check
  against) — flagged as a candidate future task, not assumed to be a bug
  without live-corpus confirmation.
  Branch `agent/citation-authority-chain` created off `dev`. `task.md`
  committed there (`9356ab3`, author `Prakash Basnet`). Assigned to Pi.
  Awaiting Prakash to dispatch.

- **AGENT-27 (2026-09-03, MERGED)**: Pi returned `78da5bb` — real commit on
  the correct branch, clean working tree (no repeat of the
  uncommitted-diff failure mode from AGENT-15/19/20/22). File scope exactly
  matched `task.md`'s allowed list (`validation_gate.py`,
  `gated_orchestrator.py` limited to `_structured_claims`'s prompt and
  `_extractive_claim`, `query_graph.py` limited to `validate_node`'s
  field-copy loop, `tests/test_validation_gate.py`) — no `_citation()`
  touch, no composer touch, no precedent touch. `_normalize()` (NFC +
  Devanagari-digit-fold + whitespace-collapse-via-`.split()`) and
  `_claim_supported()` (15-char floor + substring check) added as a third
  local copy of the existing normalization pattern, per the brief's Ponytail
  instruction — not a new shared-utils module. Wired into the existing
  `ok = ok and ...` chain in `validate_and_render()` using the
  already-fetched `expr[0]`, no second query. Verified the short-circuit
  claim myself by reading the code, not just trusting the report: because
  Python's `and` short-circuits before evaluating the right operand at all,
  `_claim_supported(...)` and its argument expressions never execute when
  `ok` is already `False` from an earlier failed check — no risk of
  `expr[0] if expr else ""` raising when `expr` is `None`, and termination/
  eligibility/hash failures reach abstention exactly as before. Extractive
  fallback self-quotes (`quote = text_ne[:300]`, same slice as `claim`) —
  correctly definitional-verbatim, confirmed no new DB round-trip added.
  `query_graph.py`'s field-copy loop gained `"quote"` alongside the existing
  `issue`/`applicability`/`condition` fields, so it survives onto the
  rendered result rather than being silently dropped.
  Test rewrite is substantively better than the prior file, not just
  updated for the new field: introduced `_passing_stubs`/`_render` helpers
  to cut duplication across the pre-existing three tests, then rewrote all
  three to carry a **genuine substring quote** rather than an absent one —
  each asserts abstention still comes from the *original* failure reason
  (bad hash, termination) with a comment saying so explicitly, directly
  proving the short-circuit claim rather than leaving it implicit. Five new
  tests, each behavioral rather than canned: exact-substring pass,
  no-match-anywhere abstain, digit-script+whitespace normalization
  (ASCII digits/collapsed newlines in the quote vs. Devanagari digits in the
  chunk — proves normalization is doing real work), empty/missing/
  whitespace-only quote abstain (parametrized over three cases), and the
  15-char floor abstaining a claim whose quote is a *real* substring
  (`assert quote in _CHUNK_TEXT` inline) but too short — the exact "floor
  matters independently of substring-match" case the brief asked for.
  Independently re-verified rather than trusting the report: `make test`
  (201 passed, 3 skipped — matches), `make lint` clean (ruff + ruff format +
  mypy --strict), `make eval-gates` against the live DB —
  `repealed-as-current: 0`, `not-yet-effective-as-current: 0`,
  `overruled-as-good-law: 0`, all matching. Independently confirmed the
  self-review's "only two claim-construction paths feed
  `validate_and_render`" claim by grepping every `"claim":` dict-literal
  site in `app/` — `gated_orchestrator.py`'s two (`_extractive_claim`,
  `_structured_claims`' prompt schema) both now emit `quote`;
  `phase_a_slice.py`'s own `claims` JSON schema is a separate ragas
  faithfulness/relevancy eval slice that never calls `validate_and_render`
  at all (confirmed via `grep` on call sites — only `query_graph.py:284`
  calls it), so it correctly wasn't in scope and doesn't need a `quote`
  field. `_citation()` confirmed untouched by diff.
  Bounded regenerate (also named in §7.7's chain) was correctly left out of
  scope per the brief — claim-support failure abstains directly, same as
  every other failure mode; regenerate remains a separate future task if
  ever built. Merged `agent/claim-support-verbatim-quote` → `dev`
  (`--no-ff`, `c1304cf`).

- **AGENT-27 scoping (2026-09-03)**: session-resume found an untracked file,
  `docs/legal_rag_ingestion_best_practices.md` — Prakash's own notes on a
  real, separate problem (`LawsChunker` mis-chunking tariff/customs schedule
  Acts like `भन्सार_महसुल_ऐन_२०८१` as prose instead of structured tariff
  rows). Not referenced anywhere in this file or any commit; not part of the
  6-task program. Flagged to Prakash before proceeding — his call: leave it
  as untracked reference notes for now, proceed with AGENT-27 as already
  queued. Not committed, not acted on; still sitting untracked in the working
  tree, revisit later if Prakash prioritizes it.
  Read `system-design.md` §2 (Core Invariants) + §7.7 (validation gate
  sequence: "exact-quote check → claim-support check → bounded regenerate →
  fallback/abstain → render citations") + §14 PS-7 before scoping, per the
  Mandatory Inputs / Design Gate. Confirmed `_extractive_claim`'s `text_ne`
  is literally `chunks.chunk_text` (`postgres_retriever.py:122`) — the
  extractive fallback path is definitionally verbatim, so its `quote` is
  just its own `claim` value, no extra query needed. Confirmed the existing
  NFC + Devanagari-digit-fold normalization pattern is already duplicated
  twice in the codebase (`pipeline.py::_content_hash`,
  `pii_redactor.py::_digit_fold`) — instructed a third small local copy in
  `validation_gate.py` rather than a new shared-utils module (Ponytail).
  One design call made directly rather than round-tripped: a **15-character
  minimum** on the normalized/stripped quote before it's eligible to
  substring-match, to close an obvious gaming vector (a single common word
  trivially "supporting" any claim) — flagged in `task.md` as tunable, not
  proven-optimal, revisit once eval data exists. Explicitly scoped bounded
  regenerate (also named in §7.7's chain) **out** of this task — abstain
  directly on claim-support failure, same as every other failure mode in the
  existing chain; regenerate is a separate future task if ever built.
  Branch `agent/claim-support-verbatim-quote` created off `dev` (fast-forward
  clean, no divergence from local `dev` at the time of branching — local
  `dev` itself is 217 commits ahead of `origin/dev`, unpushed, unrelated to
  this task). `task.md` committed there (`674ff5f`, author `Prakash
  Basnet`). Assigned to Pi (narrow, correctness-heavy gate-logic change,
  not Kimi's profile). Awaiting Prakash to dispatch.

- **AGENT-26 (2026-09-02, MERGED)**: Pi's diff was sitting **uncommitted** in
  the working tree on the correct branch (same recurring failure mode as
  AGENT-15/19/20/22 — worth a standing fix to how engineers are told to
  finish, not just noting it again). Content was correct; committed it
  myself (`51e2073`, author `Prakash Basnet`) after full review.
  `eligible_chunk_ids()` now checks `(component_uri IS NOT NULL AND
  is_eligible(c.component_uri, as_of)) OR (component_uri IS NULL AND
  effective_date_ad <= as_of)` — canonical predicate for linked chunks,
  documented fallback for unlinked ones, exactly matching `task.md`'s
  acceptance criteria. `_terminated_before()` now literally calls
  `is_eligible()` instead of re-deriving its own repeal/expiry/suspend list
  — zero duplicated predicate logic left in Python.
  **Went beyond what Pi reported to independently verify no regression**:
  ran a live-DB query myself comparing the exact old-SQL eligible-chunk-id
  count against the new predicate's count at `as_of=today` — **11,904 =
  11,904, zero drift** — before trusting the change was safe. Also queried
  `lifecycle_effect` directly and confirmed the corpus currently holds
  **zero** approved `repeal`/`expiry`/`suspend`/`declared_invalid` rows of
  any kind — meaning this fix has no live behavioral effect *today*, but
  correctly closes the gap the moment AGENT-16/18-style repeal/amend
  extraction produces an approved terminating effect. Also independently
  confirmed zero linked, previously-eligible chunks lack an approved
  `commence` effect (the specific regression risk `task.md` flagged as the
  reason for its live-corpus-coverage grounding requirement — a component
  with no `commence` row at all would flip from eligible to ineligible under
  the new predicate; confirmed this doesn't happen for anything currently
  served). Reported coverage matches and is sound: 20,977/21,504 (97.55%)
  approved act-chunks carry `component_uri`; the entire `act`-level tier
  (345 chunks, whole-document-level, not per-provision) is 0% linked by
  design — reasonable, no per-provision lifecycle applies to it — while
  `section`/`subsection`/`proviso` are all ~97-99% linked, with the small
  unlinked remainder correctly falling back to `effective_date_ad`, not
  silently dropped. Test rewrite is a genuine improvement, not just
  satisfying the brief: `FilteringCursor._component_eligible` now actually
  evaluates commence/termination/`commencement_dependency` against seeded
  `effects` data (mirroring what Postgres would compute calling
  `is_eligible()` per row) instead of string-only SQL assertions; new tests
  directly prove each acceptance-criterion scenario (`declared_invalid`
  pre-retrieval exclusion, `suspend` pre-retrieval exclusion, not-yet-
  commenced exclusion, pending-`commencement_dependency` exclusion,
  unlinked-chunk fallback with old/future/null dates). The two existing
  `_terminated_before` per-claim-as-of-direction tests
  (`test_terminated_before_true_when_repeal_on_or_before_as_of` /
  `..._false_when_repeal_strictly_after_as_of`, Core Invariant #6) were
  preserved intact through the rewrite, still exercising real predicate
  evaluation via `TerminationCursor`, not weakened. File scope exactly
  matched `task.md`'s allowed list — no `_citation()`, no migration, no
  retrieval/composer/precedent touches. Independently re-ran everything
  myself rather than trusting the report: `make test` (196 passed, 3
  skipped — matches), `make lint` clean, `make eval-gates` against the
  **live DB** (not the offline stub) — `repealed-as-current: 0`,
  `not-yet-effective-as-current: 0`, `overruled-as-good-law: 0`. Merged
  `agent/canonical-eligibility-gate` → `dev` (`--no-ff`, `66ad5e9`).

- **Retrieval-quality review + 6-task program (2026-09-02)**: Prakash pasted
  an external review rating live retrieval 4/10 for production legal use,
  with 8 numbered blockers, asking to work toward 9/10. Independently
  verified every claim against the actual code before accepting any of it
  (not from the review's word) — file:line for each:
  - **#1 eligibility gate too weak — CONFIRMED, but not the fix the review
    proposed.** `eligible_chunk_ids()` (`eligibility_gate.py:16`) only checks
    `ingestion_status='approved'` + `source_type<>'nkp_case'` +
    `effective_date_ad<=as_of` (a denormalized cache column). But
    `migrations/001_bitemporal_schema.sql:74` + `migrations/
    002_gate_suspend_fix.sql` already define a **correct** canonical
    `is_eligible(component_uri, as_of)` SQL function (commence-check +
    excludes repeal/expiry/suspend/declared_invalid) — called **nowhere in
    the live path**, only from `app/eval/gates.py:64` (the offline
    zero-tolerance scorer, tested against synthetic insert/rollback data,
    never against real retrieval traffic). The fix is "wire the existing
    predicate in," not "write a new one."
  - **#2 repeal/expiry checks happen only at validation, not pre-retrieval
    — CONFIRMED**, direct consequence of #1.
  - **New finding, not in the review**: `validate_and_render`'s
    `_terminated_before()` (`validation_gate.py:67`) is a **second,
    independently-drifted reimplementation** of the same predicate — checks
    `repeal/expiry/suspend` but is missing `declared_invalid` entirely.
    Concretely: **a court `declared_invalid` ruling is unenforceable
    anywhere in the live query path today** — not pre-retrieval, not at
    validation — despite the schema, the canonical SQL function, and the
    eval gate all modeling it correctly. This means "eval-gates 0/0/0" has
    been true and consistently reported across every merge in this file's
    history, but never actually proved the live path enforces these
    invariants — it proves the isolated SQL functions are correct, which is
    a materially weaker claim.
  - **#3 no claim-support check — CONFIRMED.** `validate_and_render`
    (`validation_gate.py:87`) checks span-hash + eligibility +
    `_terminated_before`, never that `claim.claim` text is actually
    entailed by `chunk_text`. Biggest single remaining gap after #1/#2.
  - **#4 citations are chunk metadata, not an authority chain — CONFIRMED.**
    `_citation()` (`validation_gate.py:12`) reads `act_name`/`case_id`/
    `source_publication.kind`/`ocr_confidence` off the chunk directly — no
    `component`→`expression`→amending-`lifecycle_effect` resolution, no
    `derived` labeling. PS-3 gap as described.
  - **#5 composer output not re-validated — CONFIRMED.**
    `answer_composer_node`→`_compose_answer()` (`gated_orchestrator.py:175`)
    is a second independent Gemini call whose JSON becomes `_response`
    directly (`query_graph.py:381`) with no check that its `citation` fields
    still match what `validate_and_render` actually approved.
  - **#6 no exact दफा/धारा/URI lookup path — CONFIRMED.**
    `retrieve_postgres` (`postgres_retriever.py:133`) is purely
    `vector_search` + `lexical_search` (`plainto_tsquery` bag-of-words) — no
    structured Act-title+section parse-and-match.
  - **#7 precedent not wired into `/ask` — CONFIRMED, and correct per
    design, not a bug.** `retrieve_precedent()` is called only from
    `app/eval/phase_d_slice.py:43`. `system-design.md:192`: "Case law
    answers ship [at Phase D], not before." The `phase-d/precedent` branch
    (schema/retriever/gate/eval) is already merged to `dev`, but Phase D was
    never declared active. Asked Prakash explicitly — confirmed: leave
    unwired, don't fold into this program.
  - **#8 stress suite empty — CONFIRMED.** `make stress` → literal
    `"no stress cases yet"` (Makefile:26-27).
  - Two of the review's asks were **not** turned into tasks: jurisdiction
    filtering (`work.jurisdiction` is `'NP'` on every row today — a no-op
    filter until multi-jurisdiction is real) and ACL (no schema concept
    exists anywhere — this is a public legal-QA product with no per-user
    document permissions; treated as review scope creep unless Prakash says
    otherwise later).
  - Three design questions asked and answered before scoping (not assumed):
    (a) does `suspend` terminate eligibility during its window — **yes**,
    Prakash's call, matches `validation_gate.py`'s existing (partial)
    behavior and migration 002's already-shipped fix; (b) claim-support
    mechanism — **deterministic verbatim-quote substring check**, no NLI/new
    dependency, per Prakash's call and AGENTS.md's skeleton-first
    philosophy; (c) is Phase D active — **no**, per Prakash's call, precedent
    stays unwired.
  - **Program, priority-ordered, PS-mapped, sequenced by file overlap**:
    - **AGENT-26** (dispatched now) — canonical predicate, wire
      `eligible_chunk_ids()` + `_terminated_before()` to `is_eligible()`,
      single source. Files: `eligibility_gate.py`,
      `validation_gate.py::_terminated_before` only. CI #1/#2, PS-2/PS-4/
      PS-15. Foundational, unblocks nothing else structurally but shares
      `validation_gate.py` with AGENT-27/28 so those are sequenced after.
    - **AGENT-27** (queued) — claim-support verbatim-quote check.
      `validation_gate.py`, `gated_orchestrator.py::_structured_claims`
      prompt, `query_graph.py` claim shape. Sequenced after AGENT-26 (same
      file, avoid collision per the AGENT-19/20 lesson).
    - **AGENT-28** (queued) — citation authority-chain rendering (PS-3).
      `validation_gate.py::_citation`. Sequenced after AGENT-26/27.
    - **AGENT-29** (queued, parallel-safe) — composer output
      re-validation guardrail. `gated_orchestrator.py::_compose_answer`,
      `query_graph.py::answer_composer_node`. Zero file overlap with 26-28.
    - **AGENT-30** (queued, parallel-safe) — exact दफा/धारा/उपदफा/अनुसूची/
      Act-title lookup merged into `retrieve_postgres` via existing `_rrf`.
      `postgres_retriever.py` only. Zero file overlap with 26-29.
    - **AGENT-31** (queued, run last) — real stress/red-team suite
      (repealed/current, not-yet-effective, romanized, cross-ref, proviso,
      enabling-power cells). `Makefile`, new `tests/stress/`. PS-12.
      Deliberately last — should test the corpus against the *fixed*
      invariants, not the current gaps.
    Asked Prakash whether to run AGENT-29/30 in parallel via Kimi given zero
    file overlap with AGENT-26 — declined, run everything sequentially
    through Pi for now.
  - The stray `scripts/ingest_laws.py` stash from session start (2026-09-02)
    was dropped on Prakash's call — confirmed functionally no-op (a blank
    line inside a multi-line `print()` f-string, no behavior change) and it
    no longer applied cleanly anyway, since AGENT-25 rewrote that exact
    section. Two stale local branches already merged into `dev`
    (`agent/amend-tag-correlation`, `agent/schedule-header-collision`) were
    also deleted as routine cleanup this session.

- **AGENT-24 review round 2 (2026-09-02, MERGED)**: Pi returned `bf5535e`
  (reported hash had garbled trailing digits beyond the real 7-char prefix
  `bf5535e` — the commit itself is real and matches the fix described, not
  treated as a red flag, just noted). Fix is structurally correct, not a
  superficial patch: `_resolve_co_retrieve_parents` now iterates
  `candidates` (which preserves the authority-tier/score order already
  established by `authority_ranker_node`) one hit at a time, `LIMIT 1`
  per-hit query, breaking once `len(additional) >= max_additional` — the
  exact shape of `_resolve_cross_refs`, the sibling pattern this was
  supposed to mirror from round 1. New test
  (`test_co_retrieve_parent_respects_hit_order_when_capped`) seeds 7
  candidates with 7 distinct eligible parents and asserts the kept 5 are
  exactly the first 5 in input order with correct `_issue_idx`
  propagation — directly proves the fix, not just re-testing the old
  cases. Independently re-verified rather than trusting the report:
  `make test` (192 passed, 3 skipped — matches), `make lint` clean, `make
  eval-gates` 0/0/0, all reproduced myself. File scope still exactly the
  files that needed the fix (`gated_orchestrator.py`,
  `test_co_retrieve_parent.py`). Merged `agent/co-retrieve-parent-context`
  → `dev` (`--no-ff`).

- **AGENT-24 review round 1 (2026-09-02)**: Pi returned `ac9465f` —
  `_resolve_co_retrieve_parents` in `gated_orchestrator.py`, new
  `co_retrieve_parent_resolver_node` wired into `query_graph.py` between
  `authority_ranker` and `cross_ref_resolver`, `tests/test_co_retrieve_parent.py`.
  Independently verified: file scope exactly matched `task.md` (3 files, no
  chunker/schema/eligibility-gate/validation-gate touches); graph edges
  correct; `make test` (191 passed, 3 skipped — matches), `make lint`
  clean, `make eval-gates` 0/0/0 — all reproduced myself. Test quality is
  genuinely good — all 5 required cases present, using a real behavioral
  stub cursor (`CoRetrieveCursor` actually evaluates `hit_ids`/`eligible`/
  `limit` against seeded data, not string-only SQL assertions), including
  the "already `co_retrieved` → zero queries executed" case.
  **One finding, sent back rather than merged**: `_resolve_co_retrieve_parents`'s
  SQL (`gated_orchestrator.py:414`) has no `ORDER BY` before `LIMIT
  %(limit)s`. Traced that `state["all_hits"]` is already sorted by
  `(tier, -score)` at this point in the graph (`authority_ranker_node` runs
  immediately before this resolver) — the single batched query joins all
  candidate hit_ids at once and lets Postgres pick whichever 5 matching
  rows it likes, discarding that priority ordering entirely. With
  `MAX_SUBQUERIES=3` issues × `k=5` hits each, more than 5 co-retrieve-
  eligible candidates in one query is a realistic case, not a hypothetical
  — a high-tier proviso could silently lose its operative-clause context
  to an arbitrary lower-tier one, non-deterministically, undermining PS-16's
  "always fetch" language for the dropped ones. The sibling functions this
  task was explicitly told to mirror (`_resolve_cross_refs`,
  `_fetch_enabling_chunk`) avoid exactly this by iterating hits in their
  existing rank order and stopping once the cap is hit — this
  implementation diverges from that pattern in the one place the
  divergence matters. Rework note appended to `task.md` (`b911318`) asking
  for priority-preserving truncation plus a test proving it (seed >5
  candidates in known priority order, assert the kept ones are the
  highest-priority, not just "some 5 of them"). Same branch, same
  engineer, per the Rework Loop — not re-scoped.

- **AGENT-25 (2026-09-02, MERGED)**: Pi returned `a183188` — coverage
  report, two real-corpus tests, reconstruction-rule doc. Independently
  verified rather than trusting the report: file scope matched `task.md`
  exactly (only `scripts/ingest_laws.py`, `docs/ingestion_design.md`,
  `tests/test_ingestion_pipeline.py`, new `tests/test_ingest_laws.py`).
  Traced `_print_parent_child_coverage`'s placement by hand — the diff's
  indentation change looked alarming at a glance (looked like the "done:"
  summary print moved inside the per-record loop) but is actually correct:
  still dedented to run once after the loop, still inside the open
  `connect()` context manager so `conn` is valid when the coverage query
  runs. Re-ran the chunker myself against both real `laws.jsonl` records
  independent of the diff, not just re-running the test file: every
  asserted value matched exactly (सुशासन_ऐन_२०६४'s दफा १८ — 9 pieces,
  proviso `co_retrieve_parent_index` `[22, 26]`, both parents correctly
  `subsection`-level with स्पष्टीकरण text; लेखापरीक्षण_ऐन_२०७५'s दफा ८ — 2
  subsection pieces, `co_retrieve_parent_index` both `None`, positions in
  document order). Went one step further on the second record: confirmed
  it's a genuine paragraph-fallback case, not a coincidence — दफा ८ is
  enumerated with Devanagari letters (क, ख, ग…) rather than numerals, so
  `_SUBSECTION_SPLIT_RE` (which requires `[०-९]+`) genuinely finds zero
  उपदफा anchors on this real block (4,102 chars, over the 2,400 max),
  correctly forcing `_split_oversized`'s paragraph-boundary path. `make
  test` (189 passed, 3 skipped — matches), `make lint` clean (both changed
  files are in the Makefile's fixed list this time, no manual mypy/ruff
  gap). No दफा parent rows added, `co_retrieve_parent_id` population logic
  untouched, zero overlap with AGENT-24's scope. Merged
  `agent/chunk-parent-coverage` → `dev` (`--no-ff`).

- **AGENT-24/25 scoping (2026-09-02)**: Pi rated the ingestion pipeline
  7/10 and asked grounding questions about `chunks.parent_section` /
  `co_retrieve_parent_id` coverage — first phrased in a way that read as
  retrieval-ish, then Prakash re-asked as clean ingestion-only questions.
  Answered both rounds against the actual code and a live-DB query (not
  the design doc alone), which split the finding into two genuinely
  independent problems on two different subsystems:
  - **Ingestion (write side): already correct.** `co_retrieve_parent_id`
    has been populated correctly since the very first ingestion commit
    (`84f989a`, not a regression). Live-DB audit against the full ingested
    corpus (677 docs, 53,673 chunks) confirms it: `subsection` chunks are
    100% unlinked by design (17,501/17,501, matches the documented
    invariant — they're metadata-only via `parent_section`, no दफा-level
    parent row is ever created for a split दफा); `proviso` and
    `tariff_row`/`tariff_note` are ~100% linked (51/51, 5,265/5,265,
    3,807/3,807). No reingestion needed for anything in this scope.
  - **Retrieval (read side): a real, live PS-16 gap.** Nothing in
    `app/retrieval/*` ever reads `co_retrieve_parent_id` —
    `grep -rn "co_retrieve_parent_id" --include="*.py" .` matches only
    `app/ingestion/pgvector_indexer.py`. `system-design.md` PS-16 and
    `docs/ingestion_design.md:56` explicitly require query-side context
    assembly to fetch this chain ("eval-asserted"); no such eval or code
    exists. → **AGENT-24**.
  - Prakash then explicitly decided (asked directly, not assumed): keep
    the metadata-only invariant for subsections, do **not** add दफा-level
    parent chunk rows — the "complete tree" alternative would be new
    ingestion structure (Ponytail-gated) for a linking anchor with no
    other reader. Confirmed the remaining 7→8.5 gap is entirely about
    measurability, not correctness: no automatic post-ingest coverage
    audit exists anywhere (`ingest_laws.py`'s summary only reports
    per-document outcome counts); the only chunker test touching this,
    `test_laws_chunker_structure` (`tests/test_ingestion_pipeline.py:65-95`),
    uses a synthetic `परीक्षण ऐन` fixture engineered to split cleanly —
    zero real-corpus coverage; the दफा-reconstruction path
    (`work_id`+`section_number` ORDER BY `chunk_index`) works today but is
    undocumented and untested. Prakash's own minimal-path-to-8.5 list (audit
    report, real-corpus regression tests, document the reconstruction rule)
    became AGENT-25 verbatim — explicitly **not** adding parent rows, and
    explicitly not touching `co_retrieve_parent_id`'s population logic or
    considering a rename (flagged as a possible future follow-up only, not
    folded in).
  Both branches created off `dev`, `task.md` committed on each, zero file
  overlap confirmed by construction (AGENT-24 scope is `app/retrieval/*`
  only; AGENT-25 scope is `app/ingestion/*`/`scripts/ingest_laws.py`/tests/
  docs only). AGENT-24's branch was pushed to `origin` before noticing no
  prior `agent/*` branch ever was — flagged to Prakash as a possible
  deviation from the established local-only pattern, left as-is pending his
  call. **Process note for future scoping**: the AGENT-24 assignment update
  to this file was mistakenly committed on the `agent/co-retrieve-parent-context`
  branch instead of `dev` — caught and corrected (reverted there, reapplied
  here) before any engineer work landed on either branch, per the standing
  convention that `task.md` lives on the task branch but `PROGRESS.md`
  updates land on `dev` directly.

- **Backfill run + dedup bug fix (2026-09-01/02, direct commits on `dev`,
  authored by Prakash himself — not a Wakil-dispatched task)**: running the
  AGENT-22 entry's pending backfills against the real 345-law corpus
  surfaced a real bug: `upsert_expression`'s dedup key was
  `(component_uri, as_of, text_hash)`, so identical text reprocessed on a
  later run date looked "new" — 14,963 duplicate rows written across
  14,560 components on this run alone. Fixed (`afea041`) by dropping
  `as_of` from the dedup key — identical `text_hash` means nothing
  changed, regardless of run date — with a new test file
  (`tests/test_authority_writer.py`). Separately (`14c0479`), added
  `scripts/refresh_chunk_effective_dates.py`: `chunks.effective_date_ad`
  was only ever written once at ingest time from whatever `commence`
  effects were approved *then*; approving effects afterward never
  refreshed it, so a corpus ingested before any approvals existed stayed
  permanently non-retrievable even after full sign-off. New script
  recomputes it in bulk from current lifecycle-effect approval state.
  This closes out the "still pending on Prakash" line from the AGENT-22
  re-dispatch entry below (`backfill_authority_layer.py` dry-run then
  real, plus AGENT-23's two backfill scripts) — not independently
  re-verified by Claude since these are Prakash's own direct commits, not
  an engineer dispatch under Wakil-G review.
  **Not yet run**: `scripts/refresh_chunk_effective_dates.py` itself —
  it's new, untested against the live corpus as of this session.

- **Open item found at session-resume reconciliation (2026-09-02)**:
  working tree has an unexplained unstaged hunk in `scripts/ingest_laws.py`
  — a single blank line inserted mid-`print()` f-string concatenation,
  no functional change, no attribution. Doesn't match any in-flight task.
  Left as-is pending Prakash's call (revert vs. intentional edit-in-progress)
  rather than assumed to be noise and discarded.

- **AGENT-22 re-dispatch (2026-09-01, MERGED)**: this time Pi's own
  tooling stashed the leftover AGENT-23 dirt (`pi-save-small-safety-fixes`)
  before checking out this branch instead of letting it ride along —
  confirms the root cause really was a shared working directory across
  the two dispatches, not something deeper. Real commit this time
  (`a89234f`), correct branch, clean `git status`. Reviewed in full:
  migration matches spec exactly (`chunks.component_uri`,
  `documents.source_pub_id`); `pipeline.py` extracts a shared
  `_component_uri()` helper used by both the new population code and the
  existing `_commence_date()` (removes the duplicated URI-building this
  brief asked to deduplicate); new `_chunk_component_section()` correctly
  resolves `section`-level chunks via `section_number` and
  `subsection`/`proviso`-level chunks via `parent_section`;
  `pgvector_indexer.py` threads both new columns through, using
  `getattr(chunk, "component_uri", None)` so NKP chunks (which never get
  the attribute set) don't need their own dataclass touched.
  `validation_gate.py::_citation()` joins `documents.source_pub_id →
  source_publication.kind` with a correct NULL fallback to the old
  `chunks.source_type` behavior (verified via a dedicated test with both
  branches); new `_terminated_before()` correctly checks
  `approval_status='approved' AND effect_type IN ('repeal','expiry',
  'suspend') AND lower(legal_valid_time) <= as_of`, wired into
  `validate_and_render` before `_citation()` is ever called, short-
  circuiting cleanly when eligibility already failed. The
  `backfill_authority_layer.py` extension went beyond the brief in a good
  way — uses the parser's own canonical `component.uri` directly instead
  of reconstructing it via the shared helper (avoids any chance of drift
  between two URI-building code paths) and handles both ASCII- and
  Devanagari-digit `section_number`/`parent_section` storage forms
  defensively, a real-world subtlety the brief didn't spell out.
  **One gap found and fixed directly rather than sent back**: the
  `_terminated_before` tests named around "future effect" only mocked a
  canned boolean, never actually exercising real date-comparison logic —
  the exact per-claim-as-of direction (Core Invariant #6) the brief's
  self-review section explicitly asked to verify wasn't actually proven.
  Added `TerminationCursor`/`TerminationConn` (same
  actually-evaluate-the-predicate pattern as AGENT-20's `FilteringCursor`)
  seeding a real `legal_valid_time` and varying `as_of` across it —
  proved both directions: repeal on/before `as_of` abstains, repeal
  strictly after `as_of` does not. Also fixed a ruff-format violation in
  `backfill_authority_layer.py`'s new print line (that file isn't in the
  Makefile's fixed lint list, so `make lint` alone didn't catch it — same
  pre-existing gap noted for `review_lifecycle.py`/`review_documents.py`).
  Independently re-verified everything: `make test` (184 passed, 3
  skipped), `make lint` clean, manually ruff/mypy'd
  `backfill_authority_layer.py`, `make eval-gates` all three
  zero-tolerance gates at 0.
  Merged `agent/authority-linked-citations` → `dev` (`--no-ff`).
  **Still pending on Prakash** (no live DB in either session): run
  `scripts/backfill_authority_layer.py --dry-run` then for real, followed
  by AGENT-23's `scripts/backfill_source_kind.py` and
  `scripts/recompute_content_hashes.py` (both also still `--dry-run`
  unverified against a live corpus).

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
Awaiting Prakash's direction — all findings from the 2026-09-01 follow-up
review are now closed (AGENT-19 through 23 all merged). Nothing code-side
outstanding except two informational items, neither a task unless asked:
(1) `_fetch_enabling_chunk` (`query_graph.py`) bypasses the eligibility
gate for co-retrieved enabling provisions; (2) a document/chunk still has
no per-version link to which exact `source_publication` row's content it
reflects when a work has been re-ingested more than once with different
content (today's `source_pub_id`/`component_uri` are correct for the
common single-ingestion case).
Operational, still pending — needs a live local DB, unavailable in every
session so far (`localhost:5433 connection refused`): run, in order,
`scripts/backfill_authority_layer.py --dry-run` then for real (AGENT-22 —
populates `component_uri`/`source_pub_id` for the 345 already-ingested
documents), `scripts/backfill_source_kind.py` (AGENT-23 — relabels
pre-AGENT-20 `source_publication` rows), `scripts/recompute_content_hashes.py`
(AGENT-23 — fixes `content_hash` for the canonicalization change without
touching approval state), then `scripts/review_documents.py --list` (no
document has ever been approved through the new dual-approval path).
