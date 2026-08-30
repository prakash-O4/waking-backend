# AGENT-11 — Persist parsed law structure (components, source, expression)

## Objective

`ingest_law()` currently parses full structure via `parse_law()` — components with
URIs, text, and hashes — then calls `upsert_work()` and **discards everything
else**. The bitemporal authority tables (`component`, `source_publication`,
`expression`) stay empty for every ingested law. Per Core Invariant #1
(`AGENTS.md`) the bitemporal store is supposed to be the single authority; right
now it isn't being written to at all — only search derivatives (`documents`,
`chunks`) are populated.

`app/authority/writer.py` already has the functions needed
(`upsert_source`, `upsert_component`, `upsert_expression`) — they just aren't
called from the pipeline. This task wires them in. It does **not** touch
lifecycle proposals (commencement/amendment/repeal/expiry extraction) — that's
a separate, harder task (AGENT-12) because of the dual-approval implications
below.

## Assigned branch

`agent/persist-law-structure` (base: `dev` at current HEAD, commit 836bd75)

## Acceptance criteria

1. For every successfully-ingested law record (act or regulation, tariff or
   not), `ingest_law()` writes:
   - one `source_publication` row (via `upsert_source`)
   - one `component` row per `law.components` entry (via `upsert_component`)
   - one `expression` row per component (via `upsert_expression`), with
     `as_of = date.today()` (see Step 2 — do not use `law.enactment_ad` or
     any other date; rationale below).
2. These writes happen **after VALIDATE passes** (not before — a record that
   fails the दफा-anchor check should not pollute the authority tables) and
   **before CHUNK**. Give this its own span — `_begin_span(trace,
   "PERSIST_AUTHORITY", {...})` / `_end_span(...)` — following the exact
   pattern every other stage already uses. **Do not** try to reuse the LOAD
   span: it's already closed by this point (`_end_span` called on it at the
   end of the LOAD block, before VALIDATE even begins) and the local `span`
   variable has been reassigned to VALIDATE's span by the time persistence
   runs. Reopening or re-ending an already-closed span is not a documented
   Langfuse.py pattern — don't fake metadata onto it. A new span costs
   nothing architecturally (same helper functions, same shape as every
   other stage) so this isn't a Ponytail violation. Output should include
   `component_count`.
3. If any of the three writes raises, the document is **rejected** — same
   pattern as the existing VALIDATE/CHUNK failure paths (`_set_status(...,
   "rejected")`, commit, `last_outcome = "rejected"`, `_end_span` with
   `outcome: "rejected"`, `_end_trace`, return `None`). This is **not** the
   `except Exception: log and continue` pattern used for enabling-power
   extraction or metadata enrichment — those are derivative annotations that
   may legitimately be missing; components/expressions are the authority
   record itself, so a failure to write them means the ingest did not
   actually happen and must not silently proceed to CHUNK/EMBED as if it did.
4. Re-ingesting a record with unchanged `content_hash` still produces **no**
   new component/source/expression writes (already guaranteed by the
   existing pre-trace idempotency skip at the top of `ingest_law()` — verify
   with a test, don't just assume).
5. `make test` green, `make lint` green, `make eval-gates` green (this task
   doesn't touch retrieval or gates, so the three zero-tolerance counters
   should already be at zero and stay there — if they aren't, that's a
   pre-existing issue to report, not to fix here).

## Files to modify

| Action | File |
|---|---|
| **MODIFY** | `app/ingestion/pipeline.py` — import + call the three writer functions in `ingest_law()` |
| **MODIFY** | `tests/test_ingestion_pipeline.py` — new tests (see below) |

Do not touch: `app/authority/writer.py`, `app/authority/parser.py`,
`app/ingestion/laws_chunker.py`, `app/ingestion/tariff_chunker.py`,
`app/ingestion/nkp_chunker.py`, `app/ingestion/pgvector_indexer.py`,
`app/ingestion/metadata_enricher.py`, `migrations/*.sql` (no schema change —
`component`/`source_publication`/`expression` tables already exist in
`migrations/001_bitemporal_schema.sql`), anything under
`app/ingestion/enabling_extractor.py`, `ingest_nkp_case()` (NKP cases have no
`work`/`component` — this task is laws-only).

**Do not call `insert_commence` anywhere in this task.** It's a Phase-0 stub
(`app/authority/writer.py:80-113`) that writes `lifecycle_effect` rows with
`approval_status='approved'` pre-signed by a fixed service UUID
(`PHASE0_APPROVER`). Wiring it into every ingest run would auto-approve
commencement for every दफा with zero human review — a direct violation of
Core Invariant #5 (ingestion of legal state is human-gated with dual
approval, no exceptions in code). Lifecycle proposal extraction is AGENT-12
and will need its own `approval_status='pending'` insert path, not this
stub. If you find yourself wanting to call it, stop and flag it instead.

## Step 1 — Import and wire the writer calls

In `app/ingestion/pipeline.py`, current import is:
```python
from app.authority.writer import upsert_work
```
Change to:
```python
from app.authority.writer import upsert_component, upsert_expression, upsert_source, upsert_work
```

In `ingest_law()`, right after the existing block that does:
```python
law = parse_law(record)
work_id = upsert_work(self._conn, law)
```
and **after** the VALIDATE stage passes (i.e. after the `has_dafa` check, not
before — move the persistence call down, don't duplicate `parse_law`/
`upsert_work`, which must stay in LOAD since `work_id` is needed for the
document upsert either way), add:

```python
upsert_source(self._conn, work_id, law, source_url=None)
for component in law.components:
    upsert_component(self._conn, work_id, component)
    upsert_expression(self._conn, component, as_of=date.today())
```

`date` is already imported at the top of `pipeline.py`
(`from datetime import date`).

Wrap this block in its own `PERSIST_AUTHORITY` span (`_begin_span` right
before the `upsert_source` call, `_end_span` right after the loop) — see
acceptance criterion 2 above for why this can't reuse the LOAD span. On
success, `_end_span(span, {"outcome": "passed", "component_count":
len(law.components)})`. On failure, use the exact same rejection shape as
the existing CHUNK-stage rejection block (see lines ~260-270 of the current
file for the pattern to copy: `_set_status`, `self._conn.commit()`,
`self.last_outcome = "rejected"`, `_end_span` with `outcome: "rejected"`,
print, `_end_trace`, `_flush`, `return None`).

## Step 2 — `as_of` semantics (read before implementing)

`expression.as_of` records "this text is a valid reading as of this date."
Because `upsert_expression` hardcodes `is_derived=TRUE`, every expression
this task writes is explicitly a **derived consolidation**, not authority
(§7.4) — consistent with how these documents are actually sourced (already-
consolidated act text, not a base-instrument-plus-amendments reconstruction).

Use `as_of = date.today()` — the date this reading was captured into the
system — not `law.enactment_ad` (that's when the *original* instrument was
enacted, which may predate the current consolidated text by years of
amendments) and not a fabricated "effective date" (no lifecycle proposal
exists yet to justify one — that's AGENT-12's job).

This is safe from duplicate-row accumulation: `upsert_expression`'s dedup
check is `(component_uri, as_of, text_hash)`, and this whole block only runs
when `ingest_law()` gets past the pre-trace `content_hash` skip check — i.e.
only on genuinely new/changed content, not on every re-run. Confirm this
with test 4 below rather than taking it on faith.

## Relevant design refs

- `system-design.md` §2 (Core Invariant #1: bitemporal store is single
  authority), §4 (Data model: component/source_publication/expression),
  §5 (Ingestion plane — "structure pipeline... lifecycle extraction,
  proposal only" — note lifecycle extraction is explicitly *not* in this
  task's scope), §7.4 (Consolidation is not authority — `is_derived` usage)
- `AGENTS.md` non-negotiables — Invariant #5 (human-gated dual approval) is
  why `insert_commence` is off-limits here
- `app/authority/writer.py` — `upsert_source`, `upsert_component`,
  `upsert_expression` (already implemented, do not modify signatures)
- `app/authority/parser.py` — `ParsedLaw.components: list[ParsedComponent]`,
  `ParsedComponent.text_hash` (already sha256 of `text_ne`, used by
  `upsert_expression`'s dedup check)
- `migrations/001_bitemporal_schema.sql` — `component`, `source_publication`,
  `expression` table definitions (no FK from `lifecycle_effect`/`expression`
  to `component.id` — they join by `component_uri` TEXT, not UUID; this is
  the existing convention, don't "fix" it)

## PS requirements in scope

- **PS-3** — expressions/components must be resolvable so citations can
  eventually trace to the authority chain instead of only to `chunks`. This
  task lays the data down; it does not wire retrieval/citation resolution to
  read from it (out of scope — future task).

## Zero-tolerance gates — do not touch

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

This task doesn't touch retrieval, eligibility gate, or validation gate — the
gates should be unaffected. Run `make eval-gates` anyway and report the
numbers; if any is nonzero, that's pre-existing and must be reported, not
silently fixed as a drive-by.

## Required checks

```
make test
make lint
make eval-gates
```

## Tests to write (add to `tests/test_ingestion_pipeline.py`)

Follow the existing monkeypatch pattern in this file (see
`test_langfuse_span_end_called` for the shape — `monkeypatch.setattr(pipeline_mod,
"upsert_work", fake_upsert_work)`). You'll need to add similar fakes for
`upsert_source`, `upsert_component`, `upsert_expression` to the **other**
existing tests in this file too, since they currently don't mock these and
will now hit real (unmocked) functions if you don't — check every existing
test that calls `ingest_law()` and add the new mocks so the suite doesn't
start trying real DB calls.

1. `test_ingest_law_persists_components` — a record whose content produces
   ≥2 components (per `parse_law`) → `upsert_component` called once per
   component with the correct `work_id`.
2. `test_ingest_law_persists_source` — `upsert_source` called exactly once
   with `work_id` and the parsed `law`.
3. `test_ingest_law_persists_expression_with_todays_date` — `upsert_expression`
   called with `as_of == date.today()` for each component (freeze/mock
   `date.today` if needed for determinism, or just assert the value passed
   equals `date.today()` at assertion time — pick whichever is less flaky
   and explain your choice in the commit).
4. `test_ingest_law_skip_path_no_persistence_calls` — second call with
   identical content (same `content_hash`) → none of `upsert_source`/
   `upsert_component`/`upsert_expression` called (only the pre-trace skip
   path runs).
5. `test_ingest_law_persistence_failure_rejects_document` — `upsert_component`
   (or any of the three) raises → document status set to `'rejected'`,
   `last_outcome == 'rejected'`, function returns `None`, and — critically —
   assert the CHUNK-stage chunker was **never called** (proves it didn't
   silently continue past the failure).
6. `test_ingest_law_validate_failure_before_persistence` — a record that
   fails the दफा-anchor VALIDATE check → none of the three persistence
   functions called (proves persistence happens after VALIDATE, not before).

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution of any kind.

## Return

Commit hash, changed files, checks run + results, and explicit answers to:
- Where exactly did you place the persistence block relative to the LOAD/
  VALIDATE span boundaries, and why?
- Did any existing test need its mocks updated to avoid hitting real DB
  calls, and how many?
- What were the three zero-tolerance gate numbers after your change?
- Any law record shape in the corpus where `parse_law()` produces zero
  components (e.g. a tariff-dominant document with no दफा headers at all) —
  does `upsert_source` still get called in that case, and is that correct?
