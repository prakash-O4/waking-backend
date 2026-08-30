# AGENT-12 — Commencement proposal extraction + dual-approval review CLI

## Objective

The system currently stores what a law's text *says*, but has no memory of
*when it became law*. `lifecycle_effect` (schema already exists,
`migrations/001_bitemporal_schema.sql`) is the table for this — commence /
amend / repeal / expiry / suspend / correct / declared_invalid events per
component — but nothing in ingestion writes to it. This task closes the
**commencement** slice only (not amend/repeal/expiry — see "Explicitly out
of scope" below).

This is not guesswork extraction. Every detected commencement is written as
a **pending proposal** — never auto-approved — and a small CLI is added so
a human can actually approve or reject them (right now nothing in this
codebase can flip `approval_status` to `'approved'` for any table; this CLI
is the first one). Dual approval is enforced: two distinct people must
approve before a proposal counts as law.

**Honest scope note, put here so nobody overclaims this later:** approving
a proposal through this CLI does **not** yet change what the system
retrieves or answers. `app/retrieval/eligibility_gate.py` currently derives
eligibility from `documents`/`chunks` only — it does not consult
`component`/`lifecycle_effect`/`is_eligible()` yet. Wiring the real gate to
this bitemporal layer is a separate, later task. AGENT-12 is necessary
groundwork, not a retrieval-behavior change.

## Assigned branch

`agent/lifecycle-commencement` (base: `dev` at current HEAD, commit 07dfd08)

## Corpus grounding (already done — read before writing regex)

Verified against all 677 records in `laws.jsonl` (root of repo, gitignored,
present locally). Four distinct commencement patterns, all appearing within
the first ~2500 chars of `content` (a broad `प्रारम्भ हुनेछ` search matched
654/677 docs in that window — many docs have a long amendment-history table
before reaching the actual commencement दफा, so don't shrink the search
window below ~2500 chars without re-checking):

1. **Immediate** (~90+ docs): `यो ऐन तुरुन्त प्रारम्भ हुनेछ`
2. **Relative — N days after certification** (~5+ docs, likely more —
   re-scan the full corpus yourself before finalizing, this was a narrow
   sample): `यो ऐन प्रमाणीकरण भएको एकतीसौँ दिनदेखि प्रारम्भ हुनेछ`.
   Confirmed ordinal words in the corpus: `एकतीसौँ`/`एकतिसौँ` (31),
   `आठौं` (8), `एकानब्बेऔं` (91). Effective date = `law.enactment_ad + N
   days` — `enactment_ad` is already parsed by
   `app/authority/parser.py::_enactment_ad()`.
3. **Gazette-dependent (true "pending notification")**: e.g. `नेपाल
   सरकारले नेपाल राजपत्रमा सूचना प्रकाशन गरी तोकेको मितिदेखि प्रारम्भ
   हुनेछ` (and variants: `प्रकाशित` for `प्रकाशन`, `मितिमा` for
   `मितिदेखि`, `तोकिदिएको` for `तोकेको`). This is system-design.md §7.3's
   exact case — no date exists yet, must never be fabricated.

   **Trap, already hit and avoided:** a naive regex on `राजपत्रमा सूचना
   प्रकाशन गरी तोकिएको` alone matches ~96 times across the corpus, but
   almost all of those are the boilerplate definition of "तोकिएको"
   ("prescribed") used throughout ordinary substantive sections ("...as
   prescribed by the Government by Gazette notice") — nothing to do with
   the Act's own commencement. **The regex must anchor on the whole clause
   ending in `प्रारम्भ हुनेछ`**, not just the middle phrase, or it will
   flood the review queue with garbage proposals unrelated to
   commencement.
4. **नियमावली own-publication**: `यो नियमावली नेपाल राजपत्रमा प्रकाशन
   भएको मितिदेखि प्रारम्भ हुनेछ`. This commences on the regulation's own
   Gazette publication, not a future/separate notice — closer in spirit to
   case 1 than case 3. **Caution:** checked 292 नियमावली/नियमहरू records —
   only 10 have a `enactment_ad`-parseable date marker in the first 300
   chars, because `parser.py::_enactment_ad()` only matches `प्रमाणीकरण र
   प्रकाश(न|ित) मिति`, and नियमावली headers mostly use bare `प्रकाशन
   मिति` instead (not matched). Do **not** extend `parser.py`'s regex as
   part of this task (out of scope, touches a file this task shouldn't
   modify — see below). If `law.enactment_ad` is `None` for a case-4
   match, write the proposal with `effective_date=NULL` and
   `commencement_dependency='publication_date_unknown'` — don't guess.

Any clause that doesn't cleanly match one of these four → write an explicit
`no_commencement_clause` sentinel proposal (see Step 3), same philosophy as
AGENT-10's `enabling_extractor.py` sentinel rows — never silently missing.

## Explicitly out of scope

- **Amend / repeal / expiry extraction.** Corpus check showed what looked
  like "repeal" mentions are actually entries in an amendment-history table
  at the top of each document (a numbered list of *other* acts that
  amended this one, by name/year) — not inline repeal instructions.
  Extracting real amend/repeal events needs correlating that table against
  the `<amend>` tags already stripped elsewhere in the pipeline (see
  `app/authority/parser.py::_AMEND_RE`) — a different, harder extraction
  problem that needs its own corpus study. That's a future task
  (AGENT-15), not this one.
- **Rewiring `eligibility_gate.py`** to actually consult this data (see
  honest scope note above).
- **Split/partial commencement** (different दफा commencing on different
  dates within one Act). If you find real corpus evidence of this while
  scanning, do not guess a split — write a single proposal covering the
  Act's components using whichever pattern the दफा-१/२ clause matches, and
  flag the ambiguity in `raw_clause_text` / a code comment for a human to
  notice at review time. Do not build split-commencement logic.
- Extending `app/authority/parser.py::_enactment_ad()`'s date-marker regex.
- `insert_commence()` in `writer.py` — leave it alone (still used
  elsewhere; do not modify or remove it). This task adds a **new**
  function alongside it (see Step 1).

## Files to create / modify

| Action | File |
|---|---|
| **CREATE** | `migrations/009_lifecycle_raw_clause.sql` |
| **CREATE** | `app/ingestion/commencement_extractor.py` |
| **CREATE** | `scripts/review_lifecycle.py` |
| **CREATE** | `tests/test_commencement_extractor.py` |
| **CREATE** | `tests/test_review_lifecycle.py` |
| **MODIFY** | `app/authority/writer.py` — add `propose_lifecycle_commence()` |
| **MODIFY** | `app/ingestion/pipeline.py` — new `PROPOSE_LIFECYCLE` span; capture `upsert_source()`'s return value (currently discarded) |
| **MODIFY** | `scripts/migrate.py` — register migration 009 |

Do not touch: `app/authority/parser.py`, `app/authority/models.py`,
`app/retrieval/eligibility_gate.py`, `app/ingestion/enabling_extractor.py`,
`app/ingestion/laws_chunker.py`, `app/ingestion/tariff_chunker.py`,
anything in `app/retrieval/`, `insert_commence()` (read-only reference).

## Step 1 — Migration + writer function

`migrations/009_lifecycle_raw_clause.sql`:
```sql
ALTER TABLE lifecycle_effect ADD COLUMN IF NOT EXISTS raw_clause_text TEXT;
```
Register it in `scripts/migrate.py` following the exact pattern used for
`008_work_relations` (idempotency probe + entry).

In `app/authority/writer.py`, add a new function — do not touch
`insert_commence`:

```python
def propose_lifecycle_commence(
    conn: connection,
    component_uri: str,
    source_pub_id: str,
    *,
    effective_date: date | None,
    commencement_dependency: str | None,
    raw_clause_text: str,
) -> None:
    """Write a pending commencement proposal. NEVER sets approval_status
    to anything but 'pending' — approval is exclusively the job of
    scripts/review_lifecycle.py (dual sign-off, PS-2/Invariant #5)."""
```

Dedup before insert: `SELECT 1 FROM lifecycle_effect WHERE
component_uri=%s AND effect_type='commence' AND approval_status='pending'
LIMIT 1` — skip if found (idempotent on re-ingest; don't pile up duplicate
open proposals for the same component). Note this only checks `pending` —
if an `approved` row already exists but re-ingested text now looks
different, still propose a new pending row; a human decides whether it's a
real correction (the `no_overlap` EXCLUDE constraint on `lifecycle_effect`
will catch a genuinely conflicting approval later, at approval time, not
extraction time — don't try to pre-empt that here).

`legal_valid_time`:
- `effective_date` known → `tstzrange(effective_date, NULL)` (open-ended
  forward from that date — matches `insert_commence`'s existing
  convention).
- `effective_date` is `None` (gazette-pending, or unparsed relative-day
  word, or unknown नियमावली publication date) → `'empty'::tstzrange`. This
  is deliberate: an empty range contains no points in time, correctly
  representing "not yet valid for any as-of." Do not use an
  open/unbounded range here — that would silently assert validity from
  the beginning of time, exactly the failure §7.3 exists to prevent.

`transaction_time`: `tstzrange(now(), NULL)`.

## Step 2 — `app/ingestion/commencement_extractor.py`

New module, regex-only (no LLM), following the style of
`app/ingestion/enabling_extractor.py` (deterministic, sentinel rows on
no-match, exception-guarded caller).

Build the four-pattern classifier from the corpus evidence above. For
pattern 2, build a Devanagari-ordinal-word → integer table. Requirements:
- Cover at minimum the confirmed values (एकतीसौँ/एकतिसौँ=31, आठौं=8,
  एकानब्बेऔं=91) plus enough of the 1–100 range to be useful (delays are
  always small numbers of days). **Re-scan the full corpus yourself** for
  every relative-day match before finalizing the table — the sample above
  was narrow (5 hits in a 900-char window; the true count in a 2500-char
  window is likely higher and may surface more ordinal words to verify).
- **If a parsed ordinal word isn't in the table, do not guess or
  interpolate.** Treat it the same as the gazette-pending case: write the
  proposal with `effective_date=NULL`,
  `commencement_dependency='unparsed_relative_delay'`, and the full
  matched clause in `raw_clause_text` so a human can compute it by hand.
  A wrong guess here is exactly the failure this project exists to
  prevent — an unrecognized word must fail loud (visible pending flag),
  never silently compute a wrong date.

Main entry point should take the parsed `law` (for `law.uri`,
`law.enactment_ad`, `law.components`), the raw `content`, `source_pub_id`,
and `conn`, classify the commencement clause once, then call
`propose_lifecycle_commence()` once per component in `law.components` with
identical `effective_date`/`commencement_dependency`/`raw_clause_text`
(matches how `pipeline.py::_commence_date()` already looks up commence
effects **per-दफा** `component_uri` — this task must produce data in the
shape that existing lookup already expects).

## Step 3 — Pipeline wiring

In `app/ingestion/pipeline.py`, add a `PROPOSE_LIFECYCLE` span **after**
`PERSIST_AUTHORITY` and **before** `CHUNK`. Capture `upsert_source()`'s
return value in the existing `PERSIST_AUTHORITY` block (currently
discarded) so it can be threaded through:

```python
source_pub_id = upsert_source(self._conn, work_id, law, source_url=None)
```

Unlike `PERSIST_AUTHORITY`, a failure here must **not** reject the
document — these are proposals, not authority data; a missed proposal
just means nobody gets prompted to review it yet, recoverable later via
the backfill pattern (see Step 5). Use the same
`except Exception: logger.warning(...); continue` shape as
`extract_enabling_clause`'s call site (AGENT-10), not the
`PERSIST_AUTHORITY` reject shape.

## Step 4 — `scripts/review_lifecycle.py`

```
scripts/review_lifecycle.py --list
scripts/review_lifecycle.py --approve <id> --by <uuid>
scripts/review_lifecycle.py --reject <id> --by <uuid>
```

**Approve is strictly dual-signoff, enforced in SQL logic, not just
convention:**
- Row not `pending` → error, no-op: `"already {status}"`.
- `approved_by_1` is `NULL` → set `approved_by_1 = <uuid>`, print
  "recorded as first approver — a second, different person must approve
  before this takes effect." Status stays `pending`.
- `approved_by_1` is set and equals the given `<uuid>` → **error**, refuse:
  the same person cannot be both approvers.
- `approved_by_1` is set, differs from `<uuid>`, `approved_by_2` is `NULL`
  → set `approved_by_2 = <uuid>`, flip `approval_status='approved'`.
  Print confirmation.
- Catch the `no_overlap` EXCLUDE constraint violation (fires on the
  `approval_status='approved'` UPDATE if this conflicts with an
  already-approved row for the same `component_uri`) and print a clear
  message — not a raw psycopg2 traceback.

**Reject is single-action** (lower risk direction — a wrongly rejected
proposal is recoverable by re-running ingestion or manually re-proposing;
a wrongly *approved* one asserts something is now law). Row not `pending`
→ error. Otherwise set `approval_status='rejected'`, and record the
rejecter in `approved_by_1` (reusing that column for single-decision
attribution on the reject path — document this reuse with a one-line
comment in the script; no new column for this).

**Bulk convenience, added beyond the literal spec — flagging this for
Prakash's awareness, not just doing it silently:** one detected
commencement clause produces one proposal *per दफा component* (Step 2), so
a 50-दफा Act generates 50 individual pending rows for a single real-world
decision. Approving those one `--approve <id>` at a time is impractical.
Add:
```
scripts/review_lifecycle.py --list-work <work_uri>
scripts/review_lifecycle.py --approve-work <work_uri> --by <uuid>
scripts/review_lifecycle.py --reject-work <work_uri> --by <uuid>
```
`--list-work` shows all pending commence rows for that work, grouped by
`raw_clause_text` so a reviewer can see it's the same clause repeated.
`--approve-work`/`--reject-work` apply the **exact same per-row dual-
approval / single-reject logic above**, looped over every row matching
`component_uri LIKE '<work_uri>/%' AND effect_type='commence' AND
approval_status='pending'` — not a new mechanism, just the same safe SQL
run in a loop. If this feels like scope creep, it can be dropped and left
to individual `--approve <id>` calls — flag this in your return notes
either way.

`--list` (no argument) groups all pending proposals by
`(work portion of component_uri, raw_clause_text)` for readability, but
still prints individual `id`s so single-row approval remains possible.

**Do not build:** a UI, auth beyond the `--by <uuid>` argument, a queue,
comments, a diff viewer, roles, or a dashboard. This is a boring CLI
wrapper around safe, parameterized SQL — same spirit as
`scripts/backfill_enabling_links.py`.

## Step 5 — No backfill script this time

Unlike AGENT-10/enabling-power-links, do **not** write a separate
`scripts/backfill_*.py` for already-ingested laws. Re-running
`scripts/ingest_laws.py` naturally re-triggers `PROPOSE_LIFECYCLE` for
every record whose content_hash is unchanged... **actually check this**:
the pre-trace idempotency skip in `ingest_law()` returns early on matching
`content_hash` before `PROPOSE_LIFECYCLE` would ever run. So already-
ingested laws will **not** get commencement proposals just by re-running
ingestion. Report this clearly in your return notes — Prakash needs to
decide whether a one-off backfill pass is wanted (e.g. a temporary flag,
or accepting that lifecycle proposals only apply going forward). Do not
silently build a workaround for this — surface it and stop.

## Relevant design refs

- `system-design.md` §5 (Ingestion plane — "lifecycle extraction, proposal
  only... dual approval mandatory for lifecycle"), §7.3 (Commencement
  dependency — the exact §7.3 "pending notification" state this task
  implements), §4 (Data model — `lifecycle_effect`)
- `AGENTS.md` Invariant #5 (human-gated dual approval, no exceptions in
  code) — why `insert_commence`'s auto-approve stub can't be reused, and
  why this CLI's dual-signoff logic is load-bearing, not decoration
- `app/authority/writer.py` — `insert_commence` (reference only, do not
  modify), `PHASE0_APPROVER` (do not use)
- `migrations/001_bitemporal_schema.sql` — `lifecycle_effect` columns,
  `no_overlap` EXCLUDE constraint, `is_eligible()` (not yet wired to the
  real gate — see honest scope note)
- `app/ingestion/pipeline.py` — `_commence_date()` (already reads
  per-दफा `component_uri` from `lifecycle_effect`; this task is what
  should eventually populate it), `PERSIST_AUTHORITY` span (AGENT-11,
  merged) for the exact span-usage pattern to follow
- `app/ingestion/enabling_extractor.py` — style precedent: deterministic
  regex, no LLM, explicit sentinel rows on no-match

## PS requirements in scope

- **PS-2** — commencement (`राजपत्रमा सूचना` dependency) forces
  `not_yet_effective` + pending-notification state + a
  `commencement_dependency` link. No fabricated effective date. This is
  the core requirement this task exists to satisfy (partially — full
  satisfaction needs the future gate-rewiring task too).
- Invariant #5 (`AGENTS.md`) — every lifecycle write from ingestion is
  `pending`; only the CLI's dual-signoff path can reach `approved`.

## Zero-tolerance gates — do not touch

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

This task doesn't touch retrieval or the eligibility gate (see honest
scope note), so these should be unaffected. Run `make eval-gates` anyway
and report the numbers.

## Required checks

```
make test
make lint
make eval-gates
```

## Tests to write

### `tests/test_commencement_extractor.py`
1. Immediate pattern → proposal per component, `effective_date =
   law.enactment_ad`, `commencement_dependency=NULL`.
2. Relative N-days pattern with a known ordinal word → correct computed
   `effective_date` (`enactment_ad + N days`).
3. Relative N-days pattern with an **unrecognized** ordinal word →
   `effective_date=NULL`, `commencement_dependency='unparsed_relative_delay'`,
   raw clause preserved — not a guess.
4. Gazette-dependent pattern → `effective_date=NULL`,
   `commencement_dependency` set, `legal_valid_time` is the empty range.
5. **False-positive guard**: content containing `राजपत्रमा सूचना प्रकाशन
   गरी तोकिएको` in an ordinary definitional section (not a commencement
   clause) → no gazette-dependent proposal created from that occurrence.
6. नियमावली own-publication pattern with `enactment_ad` present → uses it.
7. नियमावली own-publication pattern with `enactment_ad=None` →
   `effective_date=NULL`, `commencement_dependency='publication_date_unknown'`.
8. No pattern matches → `no_commencement_clause` sentinel proposal.
9. Idempotency: extractor called twice for the same law → no duplicate
   pending rows (dedup check holds).
10. One proposal written per component in `law.components`, all sharing
    the same effective_date/dependency/raw_clause_text.

### `tests/test_review_lifecycle.py`
11. First `--approve` call sets `approved_by_1`, status stays `pending`.
12. Second `--approve` call from a **different** uuid completes approval.
13. Second `--approve` call from the **same** uuid as the first → error,
    rejected, status unchanged.
14. `--approve` on an already-`approved`/`rejected` row → error, no-op.
15. `--reject` on a `pending` row → status becomes `rejected`,
    `approved_by_1` records the rejecter.
16. `no_overlap` constraint violation on the approving UPDATE → caught,
    clear error message, no raw traceback, transaction left in a sane
    state (rolled back, not half-applied).
17. `--approve-work`/`--reject-work` apply the same logic across every
    pending row for a work (if you keep this feature — see Step 4).

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution of any kind.

## Return

Commit hash, changed files, checks run + results, and explicit answers to:
- How many relative-day ordinal words does your table cover, and how many
  distinct ordinal words did you actually find scanning the full corpus
  (not just the 5-sample narrow search in this brief)?
- How many of the 677 laws matched each of the four patterns, and how
  many hit the `no_commencement_clause` sentinel?
- Did you keep the `--approve-work`/`--reject-work` bulk commands, or drop
  them? Why?
- Confirm: does re-running `scripts/ingest_laws.py` today actually create
  proposals for already-ingested laws, or does the idempotency skip
  prevent it (see Step 5)? What do you recommend Prakash do about the
  backlog of 677 already-ingested laws with zero commencement proposals?
- What were the three zero-tolerance gate numbers after your change?
