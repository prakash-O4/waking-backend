# task.md — AGENT-25: make the chunk parent-child invariant measurable and corpus-proven

## Objective
Prakash rated the ingestion pipeline 7/10. The parent-child linking design
itself is correct and intentional (`subsection` chunks carry `parent_section`
text only; `proviso` and `tariff_row`/`tariff_note` chunks carry a concrete
`co_retrieve_parent_id` — confirmed live: 17,501/17,501 subsections
correctly unlinked, 51/51 proviso + 9,072/9,072 tariff children correctly
linked). **Explicit decision: do not add दफा-level parent chunk rows.** The
gap to 8.5/10 is that this invariant is documented but never measured or
proven against real corpus structure. Three items, scoped directly per
Prakash's own list — do not expand this scope:

1. **Ingestion coverage audit.** Add a report that runs after a full
   `ingest_laws.py` corpus pass and prints parent-child coverage by level:
   total children (`parent_section IS NOT NULL`), linked children
   (`co_retrieve_parent_id IS NOT NULL` also), orphan children. Reference
   query (adapt as needed, this is the shape, not a mandate):
   ```sql
   SELECT level,
          COUNT(*) FILTER (WHERE parent_section IS NOT NULL) AS total_children,
          COUNT(*) FILTER (WHERE parent_section IS NOT NULL AND co_retrieve_parent_id IS NOT NULL) AS linked_children,
          COUNT(*) FILTER (WHERE parent_section IS NOT NULL AND co_retrieve_parent_id IS NULL) AS orphan_children
   FROM chunks GROUP BY level ORDER BY level;
   ```
   Wire this into `scripts/ingest_laws.py`'s existing end-of-run summary
   block (it already prints `pending_review=... skipped=... rejected=...
   failed=...` after the loop — add the coverage report as a follow-up
   print using the same connection). Expected/correct output shape: for
   `level='subsection'`, `orphan_children == total_children` (100% — this
   is correct, not a defect, don't treat it as one). For `level='proviso'`
   and the two `tariff_*` levels, `orphan_children` should be 0 or very
   near it — a nonzero count there is a real signal worth surfacing loudly,
   since that would mean the proviso/tariff linking (which is supposed to
   be total by construction) actually broke somewhere.

2. **Real-corpus regression tests.** The only existing chunker test,
   `test_laws_chunker_structure` in `tests/test_ingestion_pipeline.py:65-95`,
   uses a synthetic fixture (`LAW_FIXTURE`, a hand-built `परीक्षण ऐन` with a
   `FILLER` string repeated 30x to force a clean उपदफा split) — zero
   coverage against how real दफा actually look. Find real oversized-दफा
   examples in `laws.jsonl` (repo root) that exercise this path for real —
   at minimum: one real दफा that splits at उपदफा boundaries with a real
   स्पष्टीकरण/proviso (parent_section set correctly, co_retrieve_parent_index
   resolving to the correct operative subsection), and ideally one "ugly"
   case if the corpus has one (irregular उपदफा spacing, multiple
   स्पष्टीकरण blocks in one दफा, or a single उपदफा still exceeding
   `MAX_CHUNK_CHARS` falling to the paragraph-boundary fallback in
   `_split_oversized`). Don't invent structure the corpus doesn't have —
   if a case type genuinely isn't present, say so and drop it rather than
   synthesizing it (same discipline as every prior corpus-grounding task in
   `.agent/PROGRESS.md`, e.g. AGENT-16/17's dropped items). Add these as
   new test(s) in `tests/test_ingestion_pipeline.py` (or a new
   `tests/test_law_chunker.py` if that reads cleaner — match whichever the
   existing file organization suggests), loading the real record directly
   from `laws.jsonl` the way AGENT-17's fix did (`docs/ingestion_design.md`
   references this pattern — real record content, not a re-typed excerpt).

3. **Document the reconstruction rule.** The full दफा (for a split
   subsection/proviso group) is reconstructed today via
   `SELECT * FROM chunks WHERE work_id = X AND section_number = Y ORDER BY
   chunk_index` — this works because pieces of one दफा are always inserted
   contiguously in document order (`pgvector_indexer.py:180-181`'s
   pre-generated-UUID comment already notes chunk_index ordering, but the
   reconstruction rule itself is undocumented and untested). Add a short,
   precise note to `docs/ingestion_design.md` near the existing §2.1
   parent/child section (around line 55-56) stating this reconstruction
   rule explicitly, and add one test proving it — seed/verify that ordering
   split pieces of a real दफा by `chunk_index` reproduces document order
   correctly (this can reuse the same real-corpus fixture from item 2).

## Assigned branch
`agent/chunk-parent-coverage` (already created off `dev`, checked out).

## Explicitly forbidden / out of scope
- **Do not add a दफा-level parent chunk row or any new chunk category.**
  Prakash explicitly decided against this — metadata-only for subsections
  stays. Don't revisit this decision; if you think you've found a reason
  it's wrong, stop and flag it rather than implementing around it.
- No changes to `co_retrieve_parent_id` population logic, `parent_section`
  logic, or any chunker's splitting behavior — that logic is correct and
  already verified against the live, already-ingested corpus. This task is
  reporting/testing/documentation only.
- No changes to `app/retrieval/*` — that's a separate, already-dispatched
  task (AGENT-24, different branch) closing the query-side PS-16 gap. Zero
  file overlap by design; keep it that way.
- No schema/migration changes.
- If you rename `co_retrieve_parent_id` or touch its column comment because
  the name reads as "universal parent ID" (Prakash's own observation) —
  don't. A rename is a migration + touches ingestion, retrieval (AGENT-24),
  and every existing reference; out of scope for this task. If it's worth
  doing, flag it as a separate follow-up, don't fold it in here.

## Governing references
- `docs/ingestion_design.md` §2.1 (chunking design, lines ~55-56).
- `system-design.md` §14 PS-16.
- `AGENTS.md` — "Become one with the data" (Karpathy skill 1): read the real
  दफा by hand before writing the test, don't guess the edge cases.

## Required checks
- `make test`
- `make lint`

(`make eval-gates` not required — this task touches no gate-relevant code
path; run it anyway if convenient to confirm no regression, but it's not
the focus here.)

## Self-review checklist before returning
- Does the coverage report run against the real live DB connection
  `ingest_laws.py` already holds, or did you open a redundant second
  connection? Reuse the existing one.
- Are the new tests loading real `laws.jsonl` content (verify the record ID
  you used actually exists in the file), not a re-typed/paraphrased
  version of it?
- Does the reconstruction-rule doc addition match what the code actually
  guarantees (contiguous chunk_index per दफा within one document) rather
  than overclaiming (e.g. don't imply it works across documents or without
  the `work_id` filter)?

## Commit authorship
Every commit must be authored as `Prakash Basnet <basnetprakash090@gmail.com>`
(`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`). Never
author, co-author, or attribute any commit to Claude, Anthropic, or any AI
tool. No `Co-Authored-By: Claude` trailer or similar.

## On completion
Commit and push to `agent/chunk-parent-coverage`. Report: commit hash(es),
changed files, checks run/results, the real `laws.jsonl` record ID(s) used
for the new tests, assumptions made, and remaining risks. No extra markdown
handoff files — this `task.md` is the only one.
