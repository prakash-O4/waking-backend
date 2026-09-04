# task.md — AGENT-33: expression-staleness serve/publish gate

## Objective

Close a confirmed P0 gap: approving an `amend` `lifecycle_effect` today never
updates `expression`/`chunks.chunk_text`. Nothing in the amend
propose→approve path (`propose_lifecycle_amend()`,
`review_lifecycle.py::_approve_one()`) touches text — it only writes/flips
lifecycle metadata. The schema's own `lifecycle_effect.replacement_text`
column exists for exactly this purpose and is written by nobody, read by
nobody. `system-design.md` §7.4's own CI invariant for this case
("regenerate expression from base + verified effects, diff, don't publish on
mismatch") was never implemented anywhere in the code.

Confirmed independently (not from `replacement_text`, which stays untouched
by this task — see Forbidden below): `_citation()`
(`validation_gate.py:70-92`) already builds and shows the user an
`"amendments"` list (PS-10) for every approved amend effect on a component —
but `_expression()`'s `chunk_text` for the same claim has zero relationship
to that list. The system can tell a user "this section was amended on
<date>" while quoting pre-amendment text as current law, with no
discrepancy signal. This is invisible to the existing zero-tolerance gates:
`is_eligible()` correctly excludes `amend` from eligibility (a दफा stays in
force whether amended or not — that part is correct), so
`repealed-as-current`/`not-yet-effective-as-current` and AGENT-26/31's tests
are structurally blind to this failure mode. Correct eligibility, wrong
content — a distinct harm class from what any existing gate checks.

Prakash's explicit direction (2026-09-04): build the §7.4 CI invariant, but
as a **serve/publish blocker**, not (yet) full reconstruction — if a
component has an approved amend effect the system cannot prove the current
text reflects, retrieval/validation must abstain / treat as unpublished.
**Do not auto-apply `replacement_text` this task** — no text-substitution
mechanism, no populating that column, no wiring it into anything.

## Design (locked — do not deviate; flagged questions go back to Claude, not decided solo)

A conservative, timestamp-based proxy for "was this chunk's text captured
before the system knew about a currently-binding amendment" — not a full
base+effects reconstruction (that needs `replacement_text`, out of scope).
Deliberately fails closed: may flag a chunk stale that's actually fine, but
will never let a genuinely-uncovered amendment slip through undetected on
the timestamp dimension. That asymmetry matches `AGENTS.md`'s prime
directive ("a loud refusal beats a quiet wrong answer") — keep it, don't
try to make the predicate "smarter"/more lenient.

**Critical correctness point (PS-17, retroactive amendments):** compare
against `lower(lifecycle_effect.transaction_time)` — when the system
*recorded* the amend proposal — **not** `effective_date`/
`lower(legal_valid_time)`. A retroactive amendment (effective_date in the
past, proposed/approved just now) must still flag an old chunk stale;
comparing against `effective_date` instead would silently miss exactly that
case. `transaction_time` is set to `tstzrange(now(), NULL)` at proposal
INSERT time (`propose_lifecycle_amend()`), before either approval —
correct, since even a first-approver's pending proposal marks "the system
now knows this text may be out of date" for this predicate's purposes,
though the predicate itself only fires for effects that reach
`approval_status = 'approved'`.

### 1. New migration: `migrations/012_expression_staleness_gate.sql`

One new stored SQL function, same file convention and exact style as
`is_eligible()` (`migrations/001_bitemporal_schema.sql:74-93`,
`002_gate_suspend_fix.sql:1-20`):

```sql
CREATE OR REPLACE FUNCTION is_expression_current(
    p_component_uri    TEXT,
    p_chunk_created_at  TIMESTAMPTZ,
    p_as_of             DATE
) RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM lifecycle_effect
        WHERE component_uri = p_component_uri
          AND effect_type = 'amend'
          AND approval_status = 'approved'
          AND lower(legal_valid_time) <= p_as_of::timestamptz
          AND lower(transaction_time) > p_chunk_created_at
    );
$$;
```

Do not fold this into `is_eligible()` itself — separate concern (temporal
freshness of *text*, not in-force *status* of the component), separate
function, per Ponytail (extend via a new predicate argument to a new
function, not a new code path inside the existing one — keeps
`is_eligible()`'s existing callers/tests untouched).

An amend effect with unresolved `effective_date` (`legal_valid_time =
'empty'`, per PS-2) must never trigger this: `lower('empty'::tstzrange)` is
`NULL` in Postgres, and `NULL <= p_as_of::timestamptz` is `NULL`/falsy, so
the `EXISTS` correctly won't match — confirm this with a live/integration
check, don't just assume it.

### 2. `app/retrieval/eligibility_gate.py::eligible_chunk_ids()`

Add one additive `AND` clause to the existing query, same shape as the
existing `is_eligible(...)` call — only applies to law/regulation chunks
(`c.component_uri IS NOT NULL`), mirroring how `is_eligible()` itself is
scoped in that query:

```sql
AND (
    c.component_uri IS NULL
    OR is_expression_current(c.component_uri, c.created_at, %(as_of)s)
)
```

Combine correctly with the existing `is_eligible()` OR-branch structure —
read the current function body first
(`app/retrieval/eligibility_gate.py:16-37`) before editing; don't just
append blindly, the existing `OR` between the `component_uri IS NOT NULL`
and `component_uri IS NULL` branches must stay intact for non-law chunks.

### 3. `app/retrieval/validation_gate.py`

Add a new sibling to `_terminated_before()` (`validation_gate.py:134-136`),
e.g. `_expression_stale(conn, evidence_id, as_of)`, resolving
`component_uri` via the existing `_authority_component_uri()` helper (reuse,
don't re-derive) and the chunk's `created_at` (new column read — chunks
table already has it), then calling `is_expression_current()`. Wire it into
`validate_and_render()`'s existing `ok` chain (`validation_gate.py:139+` —
read the full function before editing) as an additional condition alongside
the existing `_terminated_before` check, so a claim backed only by a stale
chunk fails validation through the **existing** abstention path — no new
response field, no new key on the citation dict. This is the per-claim,
server-side belt-and-suspenders check (Core Invariant #4) behind gate #2's
pre-retrieval filter (Core Invariant #2) — same dual-layering `is_eligible()`
already gets, not a new pattern.

### 4. `app/eval/gates.py`

New `check_stale_expression_as_current_live(conn, os_client=None) -> int`,
same independent-re-derivation style as
`check_repealed_as_current_live()`/`check_not_yet_effective_as_current_live()`
(`app/eval/gates.py:88-148`) — **must not** call `is_expression_current()` or
`eligible_chunk_ids()`; re-derive the SQL directly (join `chunks`+`documents`,
`d.ingestion_status='approved'`, `c.component_uri IS NOT NULL`, count chunks
where an approved amend effect's `lower(transaction_time) > c.created_at`
and `lower(legal_valid_time) <= today`). This is a new live-corpus
*regression check*, not one of the three formal zero-tolerance gates listed
in `AGENTS.md`/the retrieval-hardening program table — don't add it to that
table or rename existing gates; it should read/print the same way
AGENT-32's two live checks do. Wire it into whatever `make eval-gates`
already calls for those two checks (same call site, same output shape).

## Acceptance criteria / required tests

- `is_expression_current()` unit/integration coverage (via the DB fixture
  pattern already used in this repo's migration-backed tests, or a live-DB
  test if that's the existing convention for this function's siblings —
  check how `is_eligible()` itself is tested first):
  - approved amend effect with `transaction_time` after chunk `created_at`
    and `legal_valid_time` in force at `as_of` → `FALSE` (stale).
  - approved amend effect with `transaction_time` **before** chunk
    `created_at` (chunk was (re-)ingested after the amendment was already
    known) → `TRUE` (current) — proves refreshing via re-ingestion clears
    the flag with no other intervention needed.
  - **retroactive case (PS-17):** amend effect with `effective_date` in the
    past but `transaction_time` = now (proposed/approved today) → still
    `FALSE` for an old chunk — proves the transaction_time comparator, not
    effective_date, is what's actually wired in.
  - amend effect still `pending`, or approved with unresolved
    `effective_date` (empty `legal_valid_time`) → `TRUE` (never flags
    pending/unresolved amendments, per PS-2).
- `eligible_chunk_ids()`: new test proving a chunk with a qualifying stale
  amend effect is excluded from the returned set; a matching test proving a
  properly-refreshed chunk (transaction_time before created_at) is **not**
  excluded — don't let this regress into "amend always excludes."
- `validate_and_render()`: new test proving a claim whose only evidence is a
  stale chunk gets abstained/excluded through the existing abstain path
  (mirror the existing `_terminated_before`-driven test's shape), and a
  matching test proving a properly-refreshed chunk's claim validates
  normally.
- `test_eval_gates.py`: `test_live_stale_expression_gate_is_independent_sql`
  (same `inspect.getsource()` pattern as
  `test_live_repealed_gate_is_independent_sql`/
  `test_live_pending_gate_is_independent_sql`, lines 35-48), plus a live-DB
  assertion the new check returns `0` against the real corpus (same as the
  existing `test_live_gates_return_counts`).
- `make test`, `make lint` green.
- `make eval-gates`: all three existing zero-tolerance gates still `0`
  (unaffected — this task is additive, doesn't touch `is_eligible()`), plus
  the two existing live checks from AGENT-32 still `0`, plus the new
  `check_stale_expression_as_current_live()` at `0` against the live corpus.

## Relevant System Design / PS references

- `system-design.md` §2 Core Invariants #2 (eligibility gate, pre-retrieval,
  every branch), #4 (server-side validation gate, no bypass), #6 (as-of is
  per-claim), #7 (abstention is server-owned), #10 (provenance propagates).
- `system-design.md` §7.4 (the CI reconstruct-or-don't-publish invariant this
  task implements a conservative proxy for).
- PS-2 (commencement_dependency / unresolved effective_date must never
  affect anything — verify the `empty` range case explicitly).
- PS-3 (citations resolve to the authoritative chain; this task doesn't
  change `_citation()`'s amendment-chain rendering, only adds a gate
  upstream of it).
- PS-6 (as-of is per-claim — `is_expression_current()` takes `as_of` per
  call; it must be exercised at each claim's own declared as-of inside
  `validate_and_render()`, not a session-level shortcut).
- PS-7 (abstention stays server-owned — this is a new *server-side* reason
  to abstain, going through the existing mechanism).
- PS-17 (retroactive amendments — the `transaction_time`-not-`effective_date`
  comparator; a dedicated test is required, see above).

## Allowed scope

- `migrations/012_expression_staleness_gate.sql` (new)
- `app/retrieval/eligibility_gate.py`
- `app/retrieval/validation_gate.py`
- `app/eval/gates.py`
- `tests/test_eligibility_gate.py`
- `tests/test_validation_gate.py`
- `tests/test_eval_gates.py`

No other files. If `make eval-gates`'s existing call site for the two
AGENT-32 live checks lives somewhere else (a script, a Makefile target) and
needs a matching one-line addition for the new check, that's in scope too —
report it if so, don't silently expand scope beyond a one-line addition.

## Explicitly forbidden

- Populating, reading, or wiring `lifecycle_effect.replacement_text`
  anywhere. No text-substitution/auto-patch mechanism. If it looks like the
  task needs it to be "really" correct — it does, that's deliberately
  deferred, not this task.
- Modifying `is_eligible()`'s existing signature, body, or callers.
- Modifying `amend_extractor.py`, `propose_lifecycle_amend()`, or anything
  in the amend-proposal extraction path.
- Modifying `documents.ingestion_status` re-approval flow or
  `ingest_law()`/`upsert_document()`'s re-ingestion behavior.
- Adding a new key to the citation response dict, a new table, a new
  service, or folding this into the formal zero-tolerance gate list in
  `AGENTS.md`.

## Commit authorship

Every commit is authored as `Prakash Basnet
<basnetprakash090@gmail.com>` — never Claude, Anthropic, or any AI tool.
No `Co-Authored-By: Claude` trailer or similar. Use
`git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
