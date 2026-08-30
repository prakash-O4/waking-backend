# AGENT-13 — Backfill authority layer for already-ingested laws

## Objective

AGENT-11 and AGENT-12 both only run inside `ingest_law()`, which the
pre-trace `content_hash` idempotency skip short-circuits for anything
already ingested with unchanged content. Checked the live local DB before
writing this brief — confirmed, not assumed:

```
documents: 345    component: 0    source_publication: 0
expression: 0     lifecycle_effect: 0    work: 345
```

All 345 already-ingested acts have a `work` row (that part predates
AGENT-11) but **zero** rows in `component`, `source_publication`,
`expression`, or `lifecycle_effect`. This task backfills all four,
one-off, for the existing corpus — not just commencement proposals
(AGENT-12's original gap), the full authority layer AGENT-11 also never
wrote for these records.

## Assigned branch

`agent/backfill-authority-layer` (base: `dev` at current HEAD, commit
e732992)

## The one thing that will silently break this if you get it wrong

`component.uri`, and therefore every `expression`/`lifecycle_effect` row
keyed off it, is built from `law.uri`, which `app/authority/parser.py::
make_uri()` computes **deterministically from `record["name"]`** (BS year
+ slug extraction). The `documents` table does **not** store `name`,
`document_type`, or `work_id` (checked `migrations/005_ingestion_pipeline
.sql` — those live on `chunks`, not `documents`). If you reconstruct a
`record` dict from partial DB columns and feed it to `parse_law()`, there
is a real risk of computing a **different** `law.uri` than the one
already stored in `work.uri` for that document — every component/
expression/commence row you write would then use a URI that
`pipeline.py::_commence_date()` will never actually query for. That's not
a crash, it's silent dead data — exactly the failure mode this project
exists to prevent.

**Do not reconstruct records from the DB.** Instead:

1. Load `laws.jsonl` (repo root, gitignored, present locally — 677
   records) once into `{source_id: record}`, where `source_id =
   str(record.get("_id") or record.get("name") or "")` — this is the
   **exact** formula `ingest_law()` uses (see `pipeline.py` line ~159),
   copy it verbatim, don't approximate it.
2. For each `documents` row with `source_type IN ('act', 'regulation')`,
   look up its `source_id` in that dict. Not found → skip, log a warning,
   count it in the summary.
3. **Safety check before trusting the match**: recompute
   `pipeline._content_hash(record["content"])` (import and reuse that
   function — don't reimplement NFC+sha256 by hand) and compare against
   `documents.content_hash`. Mismatch → skip, log a warning, count it.
   This catches corpus-file drift since original ingest; better to abstain
   than backfill authority data against text that may no longer match
   what's actually indexed.
4. `law = parse_law(record)` on the **original** record — this reproduces
   the exact `law.uri` computed at original ingest time, because it's the
   same deterministic function on the same input. No rebasing, no URI
   surgery.
5. Fetch `work_id` via `SELECT DISTINCT work_id FROM chunks WHERE
   document_id=%s LIMIT 1` (same join AGENT-10's
   `scripts/backfill_enabling_links.py::_regulation_work_ids` already
   uses — `chunks.work_id` is the only place `work_id` survives per-
   document, since `documents` doesn't carry it). No `work_id` found on
   any chunk → skip, log, count.
6. **Assert, don't just trust**: fetch `work.uri` for that `work_id` and
   compare to `law.uri` from step 4. They should always match (same
   record, same function) — but if they don't, that's a real
   inconsistency (corpus record edited, or a `make_uri()` change since
   original ingest). Skip that document, log a loud warning with both
   URIs, count it. Do not write component/expression/commence rows built
   on a `law.uri` that disagrees with the already-stored `work.uri`.

## Files to create / modify

| Action | File |
|---|---|
| **CREATE** | `scripts/backfill_authority_layer.py` |
| **CREATE** | `tests/test_backfill_authority_layer.py` |

Do not touch: `app/authority/writer.py`, `app/authority/parser.py`,
`app/ingestion/pipeline.py`, `app/ingestion/commencement_extractor.py`,
`scripts/backfill_enabling_links.py`, `scripts/ingest_laws.py`. This task
is a new script that calls existing, unmodified functions — it does not
change ingestion behavior for new records.

## Step 1 — Per-document backfill logic

For each matched, hash-verified, URI-consistent document (steps 1–6
above):

```python
source_pub_id = upsert_source(conn, work_id, law, source_url=None)
for component in law.components:
    upsert_component(conn, work_id, component)
    upsert_expression(conn, component, as_of=date.today())
extract_commencement_proposals(
    law=law, content=record["content"], source_pub_id=source_pub_id, conn=conn
)
```

Import these from `app.authority.writer` and
`app.ingestion.commencement_extractor` — do not reimplement any of this
logic. This is the same sequence `ingest_law()` runs in
`PERSIST_AUTHORITY` + `PROPOSE_LIFECYCLE`, replayed here for records that
skip those stages on normal re-ingest.

All four writer functions used here are already idempotent
(`upsert_component`: `ON CONFLICT (uri) DO NOTHING`; `upsert_expression`:
dedup on `(component_uri, as_of, text_hash)`; `upsert_source`: dedup on
`(work_id, sha256)`; `propose_lifecycle_commence`: dedup on
`(component_uri, effect_type, approval_status='pending')`) — running this
script twice must not create duplicate rows. Write a test that proves
this (see below), don't just assert it in a comment.

## Step 2 — Transaction granularity

**Commit per document, not once at the end.** With 345 (soon more)
documents in one run, a failure on document #200 must not lose the work
already committed for documents #1–199. Wrap each document's writes in
its own transaction: on success, `conn.commit()`; on any exception, `conn.
rollback()`, log the `source_id` and error, count it, continue to the
next document. Follow `ingest_law()`'s `SAVEPOINT`/`ROLLBACK TO SAVEPOINT`
pattern from `PROPOSE_LIFECYCLE` if you're inside a longer-lived
connection, or just use plain per-document `commit()`/`rollback()` on a
single connection the way `scripts/review_lifecycle.py` does — either is
fine, pick whichever is simpler given how you structure the loop, but
don't do one giant transaction for all 345 documents.

## Step 3 — `--dry-run` flag

Match the convention `scripts/ingest_laws.py`/`scripts/ingest_nkp.py`
already use (per `.agent/PROGRESS.md`'s PE-A entry: "CLI scripts with
dry-run mode"). `--dry-run` runs the full match/hash-check/parse/URI-
assert pipeline and prints what *would* be written, without calling any
of the writer functions or committing anything.

## Step 4 — Summary report

End-of-run counts, printed (`Counter`-based, matching
`backfill_enabling_links.py`'s style):
- documents processed successfully
- skipped: not found in `laws.jsonl`
- skipped: content_hash mismatch
- skipped: no `work_id` found via `chunks`
- skipped: `law.uri` != stored `work.uri` (the consistency assert)
- skipped: exception during write (with a sample of source_ids, not all — bounded output)
- components written, source_publication rows written, expression rows
  written
- commencement proposals by `commencement_dependency` value (immediate /
  relative-delay-resolved / `unparsed_relative_delay` /
  `gazette_notification_pending` / `publication_date_unknown` /
  `enactment_date_unknown` / `no_commencement_clause`)

## Relevant design refs

- `.agent/PROGRESS.md` AGENT-11 and AGENT-12 entries — what this backfills
- `scripts/backfill_enabling_links.py` (AGENT-10) — structural precedent:
  `chunks.work_id` join, `Counter`-based summary, idempotent by
  construction, `try`/`except`/`rollback`/`raise` shape at the top level
- `app/ingestion/pipeline.py` — `_content_hash()` (reuse, don't
  reimplement), `PERSIST_AUTHORITY`/`PROPOSE_LIFECYCLE` blocks (the exact
  sequence this script replays), `source_id` derivation formula (line
  ~159)
- `app/authority/parser.py` — `parse_law()`, `make_uri()` (why URI
  reconstruction from partial data is unsafe — read `make_uri()` before
  writing anything, understand what it depends on)
- `migrations/005_ingestion_pipeline.sql` — confirms `documents` has no
  `work_id`/`name`/`document_type` column (`chunks` does)

## PS requirements in scope

- **PS-3** — this backfill is what makes citation resolution to the
  authority chain possible for the existing corpus, not just newly
  ingested laws going forward.
- Invariant #1 (`AGENTS.md`) — the bitemporal store is supposed to be the
  single authority; right now it's empty for the entire existing corpus.
  This task closes that for good, not just for new ingests.
- Invariant #5 — commencement proposals written here are `pending`, same
  as live ingestion; this script calls `extract_commencement_proposals`
  unmodified, it doesn't get a different (looser) approval path just
  because it's a backfill.

## Zero-tolerance gates — do not touch

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

Run `make eval-gates` and report the numbers — this task doesn't touch
retrieval, but verify anyway.

## Required checks

```
make test
make lint
make eval-gates
```

## Tests to write

1. Document found in `laws.jsonl`, hash matches, `work_id` resolves,
   `law.uri == work.uri` → all four writer functions called with correct
   arguments.
2. `source_id` not found in `laws.jsonl` → skipped, no writer calls, counted.
3. Content hash mismatch between `laws.jsonl` record and
   `documents.content_hash` → skipped, no writer calls, counted.
4. No `work_id` resolvable via `chunks` → skipped, no writer calls, counted.
5. `law.uri` (from `parse_law()`) disagrees with stored `work.uri` →
   skipped, no writer calls, counted, warning logged with both URIs.
6. Running the backfill twice on the same document → second run makes no
   new inserts (idempotency holds end-to-end, not just per-function in
   isolation).
7. `--dry-run` → matching/checking logic runs, summary prints, but zero
   writer-function calls and zero commits.
8. A document whose failure happens mid-write (e.g. `upsert_expression`
   raises on the 2nd of 3 components) → that document's transaction rolls
   back cleanly (no partial component/expression rows for it), and the
   loop continues to the next document rather than aborting the whole run.

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution of any kind.

## Return

Commit hash, changed files, checks run + results, and explicit answers to:
- Of the 345 (or however many exist by the time you run this)
  already-ingested acts, how many were successfully backfilled vs. skipped,
  broken down by the skip reasons in Step 4?
- Did the `law.uri != work.uri` consistency assert ever actually fire?
  If yes, on which documents, and what's your best guess why?
- Component/source/expression/commencement-proposal counts after running
  for real (not `--dry-run`) against the local DB.
- Did you run this against the actual local DB (`DATABASE_URL` in `.env`)
  or only test it with mocks? If only mocks, say so explicitly — don't
  imply it ran for real if it didn't.
