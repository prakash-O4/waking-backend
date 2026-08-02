# Task PD-A — Phase D: Precedent Schema + Eval Gate + Retriever

**Engineer:** Pi  
**Branch:** `phase-d/precedent`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Wire the holding-level precedent subsystem described in `system-design.md §6`:

1. **DB schema** — three tables (`precedent`, `precedent_holding`, `precedent_relation`) plus the `is_good_law(holding_id, as_of)` SQL function that derives status at the holding level from competent-bench relations, as-of the answer date.
2. **`overruled-as-good-law = 0` eval gate** — third zero-tolerance gate wired into `app/eval/gates.py` and `make eval-gates`.
3. **Precedent retriever skeleton** — `app/retrieval/precedent_retriever.py` that queries holdings filtered by `is_good_law()`.
4. **Models** — `RelationType` enum and dataclasses in `app/authority/precedent_models.py`.

Case law answers do not ship until this gate exists and is green (PS-1). The retriever will return empty until the precedent corpus is ingested — that is correct and expected for now.

---

## Acceptance criteria

1. `migrations/004_precedent_schema.sql` creates `precedent`, `precedent_holding`, `precedent_relation` tables and the `is_good_law(uuid, date)` SQL function.
2. `is_good_law(holding_id, as_of)` returns `FALSE` when an approved, competent-bench overrule/reverse relation covers the query date. Returns `TRUE` otherwise (including per-incuriam smaller-bench overrules).
3. `check_overruled_as_good_law()` in `app/eval/gates.py` returns 0 when the gate is working correctly.
4. `make eval-gates` prints:
   ```
   repealed-as-current: 0
   not-yet-effective-as-current: 0
   overruled-as-good-law: 0
   ```
   And exits 0. Exits 1 if any value is non-zero.
5. Without `SUPABASE_DB_URL` set, `make eval-gates` still prints all three lines as `0` and exits 0 (same graceful-skip pattern as existing gates).
6. `app/retrieval/precedent_retriever.py` exports `retrieve_precedent(conn, query, as_of, k=5) -> list[dict]`, returns empty list when the DB has no precedent rows.
7. `make test` green (24 passed + new tests, 1 skipped).
8. `make lint` green (include all new files).
9. `make eval-gates` green (all three gates = 0).

---

## New file: `migrations/004_precedent_schema.sql`

```sql
-- A Supreme Court case (नजिर)
CREATE TABLE IF NOT EXISTS precedent (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uri          TEXT NOT NULL UNIQUE,    -- /np/precedent/{year}/{case_no}
    title        TEXT NOT NULL,
    decided_date DATE,
    bench_size   INT NOT NULL CHECK (bench_size > 0),
    court        TEXT NOT NULL DEFAULT 'supreme_court',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A specific holding/proposition within a case.
-- Overruling is at holding level, never at case level.
CREATE TABLE IF NOT EXISTS precedent_holding (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    precedent_id  UUID NOT NULL REFERENCES precedent(id) ON DELETE CASCADE,
    holding_text  TEXT NOT NULL,
    source_span   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Temporal relation between a source case and a target holding.
-- Human-extracted, dual-approval gated (same pattern as lifecycle_effect).
CREATE TABLE IF NOT EXISTS precedent_relation (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_case_id    UUID NOT NULL REFERENCES precedent(id),
    target_holding_id UUID NOT NULL REFERENCES precedent_holding(id),
    relation_type     TEXT NOT NULL CHECK (relation_type IN (
                          'overrules','reverses','distinguishes','affirms','questions'
                      )),
    bench_strength    INT NOT NULL CHECK (bench_strength > 0),
    legal_valid_time  TSTZRANGE NOT NULL DEFAULT tstzrange(now(), 'infinity'),
    source_span       TEXT,
    approval_status   TEXT NOT NULL DEFAULT 'pending'
                          CHECK (approval_status IN ('pending','approved','rejected')),
    approved_by_1     UUID,
    approved_by_2     UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Holding-level good-law derivation.
-- Returns FALSE if any approved overrule/reverse from a competent-or-larger bench
-- covers the query date. TRUE otherwise (including per-incuriam smaller-bench rulings).
CREATE OR REPLACE FUNCTION is_good_law(p_holding_id UUID, p_as_of DATE)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN NOT EXISTS (
        SELECT 1
        FROM precedent_relation pr
        JOIN precedent source_case  ON source_case.id = pr.source_case_id
        JOIN precedent_holding ph   ON ph.id = pr.target_holding_id
        JOIN precedent target_case  ON target_case.id = ph.precedent_id
        WHERE pr.target_holding_id = p_holding_id
          AND pr.relation_type IN ('overrules', 'reverses')
          AND source_case.bench_size >= target_case.bench_size  -- competent bench
          AND pr.approval_status = 'approved'
          AND pr.legal_valid_time @> p_as_of::TIMESTAMPTZ
    );
END;
$$ LANGUAGE plpgsql;
```

Update `scripts/migrate.py` to apply migration 004.

---

## New file: `app/authority/precedent_models.py`

```python
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from datetime import date


class RelationType(str, Enum):
    OVERRULES = "overrules"
    REVERSES = "reverses"
    DISTINGUISHES = "distinguishes"
    AFFIRMS = "affirms"
    QUESTIONS = "questions"


@dataclass
class PrecedentRelation:
    source_case_uri: str
    target_holding_id: str
    relation_type: RelationType
    bench_strength: int
    legal_valid_from: date
    source_span: str | None
    approval_status: str = "pending"
```

---

## New file: `app/retrieval/precedent_retriever.py`

```python
from __future__ import annotations
from datetime import date
from typing import Any
from psycopg2.extensions import connection


def retrieve_precedent(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    """
    ILIKE search over precedent_holding.holding_text, filtered by is_good_law().
    Returns [{holding_id, holding_text, case_uri, case_title, decided_date}].
    Returns empty list when precedent tables are empty (no corpus yet).
    """
    tokens = [t for t in query.split() if t][:5]
    if not tokens:
        return []
    where = " AND ".join(["ph.holding_text ILIKE %s"] * len(tokens))
    params = [f"%{t}%" for t in tokens]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ph.id, ph.holding_text, p.uri, p.title, p.decided_date
            FROM precedent_holding ph
            JOIN precedent p ON p.id = ph.precedent_id
            WHERE {where}
            LIMIT 50
            """,
            params,
        )
        rows = cur.fetchall()
    results: list[dict[str, Any]] = []
    for holding_id, holding_text, case_uri, case_title, decided_date in rows:
        with conn.cursor() as cur:
            cur.execute("SELECT is_good_law(%s, %s)", (holding_id, as_of))
            row = cur.fetchone()
        if row and row[0]:
            results.append({
                "holding_id": str(holding_id),
                "holding_text": holding_text,
                "case_uri": case_uri,
                "case_title": case_title,
                "decided_date": decided_date,
            })
        if len(results) >= k:
            break
    return results
```

---

## Modify: `app/eval/gates.py`

Add `check_overruled_as_good_law()` and wire it into `main()`.

### Helper pattern (follow existing _insert_effect / _delete_effect pattern):

```python
def _insert_precedent_fixture(conn: connection) -> tuple[str, str, str]:
    """
    Insert: target case (bench=3), holding, source case (bench=5, competent),
    approved overrule relation covering 2020-01-01 onward.
    Returns (target_case_id, holding_id, source_case_id, relation_id) — just the holding_id
    is needed for is_good_law(); return all four for cleanup.
    """
    # Use SAVEPOINT so test inserts don't survive on rollback.
```

### Gate function:

```python
def check_overruled_as_good_law(
    conn: connection, os_client: object | None = None
) -> int:
    """
    Inserts a test target precedent (bench=3), one holding, and a source precedent
    (bench=5) with an approved 'overrules' relation covering date(2024, 1, 1).
    Verifies is_good_law(holding_id, date(2024,1,1)) returns False.
    Returns 0 (gate works) or 1 (violation — gate did not block overruled holding).
    Uses SAVEPOINT for full rollback isolation.
    """
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT precedent_gate_check")
    try:
        # Insert target case
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent (uri, title, bench_size) VALUES (%s,%s,%s) RETURNING id",
                ("/test/target", "Target Case", 3),
            )
            target_case_id = cur.fetchone()[0]  # type: ignore[index]
        # Insert holding
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent_holding (precedent_id, holding_text) VALUES (%s,%s) RETURNING id",
                (target_case_id, "Test holding text"),
            )
            holding_id = cur.fetchone()[0]  # type: ignore[index]
        # Insert source case (larger bench — competent to overrule)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent (uri, title, bench_size) VALUES (%s,%s,%s) RETURNING id",
                ("/test/source", "Source Case", 5),
            )
            source_case_id = cur.fetchone()[0]  # type: ignore[index]
        # Insert approved overrule relation
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO precedent_relation
                    (source_case_id, target_holding_id, relation_type, bench_strength,
                     legal_valid_time, approval_status)
                VALUES (%s, %s, 'overrules', 5, '[2020-01-01,)'::tstzrange, 'approved')
                """,
                (source_case_id, holding_id),
            )
        # Gate check: is_good_law must return False for overruled holding
        with conn.cursor() as cur:
            cur.execute("SELECT is_good_law(%s, %s)", (holding_id, date(2024, 1, 1)))
            row = cur.fetchone()
        is_good = bool(row[0]) if row else True
        return int(is_good)  # 0 if gate blocks correctly, 1 if gate is broken
    finally:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT precedent_gate_check")
```

### Update `main()`:

```python
def main() -> None:
    if not os.getenv("SUPABASE_DB_URL"):
        print("repealed-as-current: 0")
        print("not-yet-effective-as-current: 0")
        print("overruled-as-good-law: 0")
        return
    with connect() as conn:
        repealed = check_repealed_as_current(conn)
        pending = check_not_yet_effective_as_current(conn)
        overruled = check_overruled_as_good_law(conn)
        conn.commit()
    print(f"repealed-as-current: {repealed}")
    print(f"not-yet-effective-as-current: {pending}")
    print(f"overruled-as-good-law: {overruled}")
    raise SystemExit(1 if repealed or pending or overruled else 0)
```

---

## New file: `tests/test_precedent_gate.py`

Mock-based — no live DB. Test the gate logic by mocking `connect` and Postgres cursor.

Cover:
1. `check_overruled_as_good_law`: mock `is_good_law` returning `False` → gate returns 0 (correct).
2. `check_overruled_as_good_law`: mock `is_good_law` returning `True` → gate returns 1 (violation detected).
3. `retrieve_precedent`: with empty DB result → returns `[]`.
4. `retrieve_precedent`: mock holding row where `is_good_law` returns `True` → result included.
5. `retrieve_precedent`: mock holding row where `is_good_law` returns `False` → result excluded.

**Important:** Because the gate uses SAVEPOINT and real INSERT/ROLLBACK, it cannot be easily unit-tested without a live DB. Write the tests against the retriever instead (items 3–5), and for the gate (items 1–2), write integration-style tests that patch `conn.cursor()` to simulate the DB responses. Follow the pattern already used in `tests/test_eligibility_gate.py` if that pattern exists, otherwise use `unittest.mock`.

---

## Makefile changes

1. Add to all three lint lines:
   - `app/authority/precedent_models.py`
   - `app/retrieval/precedent_retriever.py`
   - `tests/test_precedent_gate.py`

2. The `eval-gates` target is unchanged (it calls `python3 -m app.eval.gates` — the updated `main()` handles the rest).

---

## Do NOT touch

- `app/retrieval/gated_orchestrator.py` — do not wire precedent retriever in. The corpus doesn't exist yet; wiring happens after ingestion.
- `app/main.py` — no changes.
- `migrations/001_bitemporal_schema.sql`, `002_gate_suspend_fix.sql`, `003_bs_ad_calendar.sql`.
- `app/eval/gates.py` other than the additions described above.

---

## Forbidden

- Do not store good-law status as a boolean column on `precedent` or `precedent_holding`. It is derived at query time from relations, as-of the answer date. (PS-1)
- Do not let a smaller-bench overrule block a holding. `is_good_law` must enforce `source_case.bench_size >= target_case.bench_size`.
- No new pip dependencies.

---

## PS requirements in scope

| PS | Enforcement |
|---|---|
| PS-1 | Good-law status derived at holding level from competent-bench relations as-of answer date. Never stored as boolean. Gate: `overruled-as-good-law = 0`. |

## Zero-tolerance gates (must all stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`  ← **NEW**

---

## Required checks

```
make lint
make test
make eval-gates
```

`make eval-gates` must print all three lines and exit 0.

---

## Commit authorship

Every commit MUST use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By: Claude`, no "Generated with Claude", no AI attribution of any kind.

---

## Return to Claude (via Prakash)

- Commit hash
- Changed/created files list
- `make lint` output
- `make test` output
- `make eval-gates` output (must show all three gates)
- Assumptions and remaining risks
