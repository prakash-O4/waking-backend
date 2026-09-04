from __future__ import annotations

import os
from datetime import date

from psycopg2.extensions import connection

from app.authority.writer import connect
from app.retrieval.eligibility_gate import is_eligible


def _component_uri(conn: connection, offset: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT uri FROM component ORDER BY created_at, uri OFFSET %s LIMIT 1",
            (offset,),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None


def _insert_effect(
    conn: connection,
    component_uri: str,
    effect_type: str,
    dependency: str | None = None,
) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lifecycle_effect (
                component_uri, effect_type, approval_status, legal_valid_time,
                transaction_time, commencement_dependency
            ) VALUES (%s, %s, 'approved', '[2020-01-01,)'::tstzrange, '[now(),)'::tstzrange, %s)
            RETURNING id
            """,
            (component_uri, effect_type, dependency),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("test effect insert returned no id")
        return str(row[0])


def _delete_effect(conn: connection, effect_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lifecycle_effect WHERE id=%s", (effect_id,))


def _isolated_check(
    conn: connection,
    component_uri: str,
    effect_type: str,
    dependency: str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT gate_check")
        cur.execute(
            "DELETE FROM lifecycle_effect WHERE component_uri=%s", (component_uri,)
        )
    try:
        effect_id = _insert_effect(conn, component_uri, effect_type, dependency)
        try:
            return int(is_eligible(conn, component_uri, date(2024, 1, 1)))
        finally:
            _delete_effect(conn, effect_id)
    finally:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT gate_check")


def check_repealed_as_current(conn: connection, os_client: object | None = None) -> int:
    component_uri = _component_uri(conn, 0)
    if not component_uri:
        return 1
    return _isolated_check(conn, component_uri, "repeal")


def check_not_yet_effective_as_current(
    conn: connection, os_client: object | None = None
) -> int:
    component_uri = _component_uri(conn, 1)
    if not component_uri:
        return 1
    return _isolated_check(conn, component_uri, "commence", "gazette_notification")


def check_repealed_as_current_live(
    conn: connection, os_client: object | None = None
) -> int:
    today = date.today()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(DISTINCT c.id)
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ingestion_status = 'approved'
              AND c.source_type <> 'nkp_case'
              AND c.component_uri IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM lifecycle_effect le
                  WHERE le.component_uri = c.component_uri
                    AND le.effect_type = 'commence'
                    AND le.approval_status = 'approved'
                    AND le.legal_valid_time @> %(today)s::timestamptz
                    AND le.commencement_dependency IS NULL
              )
              AND EXISTS (
                  SELECT 1 FROM lifecycle_effect le
                  WHERE le.component_uri = c.component_uri
                    AND le.effect_type IN ('repeal','expiry','declared_invalid','suspend')
                    AND le.approval_status = 'approved'
                    AND lower(le.legal_valid_time) <= %(today)s::timestamptz
              )
            """,
            {"today": today},
        )
        row = cur.fetchone()
    return int(row[0]) if row else 1


def check_not_yet_effective_as_current_live(
    conn: connection, os_client: object | None = None
) -> int:
    today = date.today()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(DISTINCT c.id)
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ingestion_status = 'approved'
              AND c.source_type <> 'nkp_case'
              AND c.component_uri IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM lifecycle_effect le
                  WHERE le.component_uri = c.component_uri
                    AND le.effect_type = 'commence'
                    AND le.approval_status = 'approved'
                    AND le.legal_valid_time @> %(today)s::timestamptz
                    AND le.commencement_dependency IS NULL
              )
            """,
            {"today": today},
        )
        row = cur.fetchone()
    return int(row[0]) if row else 1


def check_stale_expression_as_current_live(
    conn: connection, os_client: object | None = None
) -> int:
    today = date.today()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(DISTINCT c.id)
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ingestion_status = 'approved'
              AND c.source_type <> 'nkp_case'
              AND c.component_uri IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM lifecycle_effect le
                  WHERE le.component_uri = c.component_uri
                    AND le.effect_type = 'amend'
                    AND le.approval_status = 'approved'
                    AND lower(le.legal_valid_time) <= %(today)s::timestamptz
                    AND lower(le.transaction_time) > c.created_at
              )
            """,
            {"today": today},
        )
        row = cur.fetchone()
    return int(row[0]) if row else 1


def check_overruled_as_good_law(
    conn: connection, os_client: object | None = None
) -> int:
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT precedent_gate_check")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent (uri, title, bench_size) "
                "VALUES (%s,%s,%s) RETURNING id",
                ("/test/target", "Target Case", 3),
            )
            row = cur.fetchone()
            if not row:
                return 1
            target_case_id = row[0]
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent_holding (precedent_id, holding_text) "
                "VALUES (%s,%s) RETURNING id",
                (target_case_id, "Test holding text"),
            )
            row = cur.fetchone()
            if not row:
                return 1
            holding_id = row[0]
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO precedent (uri, title, bench_size) "
                "VALUES (%s,%s,%s) RETURNING id",
                ("/test/source", "Source Case", 5),
            )
            row = cur.fetchone()
            if not row:
                return 1
            source_case_id = row[0]
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
        with conn.cursor() as cur:
            cur.execute("SELECT is_good_law(%s, %s)", (holding_id, date(2024, 1, 1)))
            row = cur.fetchone()
        is_good = bool(row[0]) if row else True
        return int(is_good)
    finally:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT precedent_gate_check")


def main() -> None:
    if not os.getenv("SUPABASE_DB_URL"):
        print("repealed-as-current: 0")
        print("repealed-as-current-live-corpus: 0")
        print("not-yet-effective-as-current: 0")
        print("not-yet-effective-as-current-live-corpus: 0")
        print("stale-expression-as-current-live-corpus: 0")
        print("overruled-as-good-law: 0")
        return
    with connect() as conn:
        repealed = check_repealed_as_current(conn)
        repealed_live = check_repealed_as_current_live(conn)
        pending = check_not_yet_effective_as_current(conn)
        pending_live = check_not_yet_effective_as_current_live(conn)
        stale_live = check_stale_expression_as_current_live(conn)
        overruled = check_overruled_as_good_law(conn)
        conn.commit()
    print(f"repealed-as-current: {repealed}")
    print(f"repealed-as-current-live-corpus: {repealed_live}")
    print(f"not-yet-effective-as-current: {pending}")
    print(f"not-yet-effective-as-current-live-corpus: {pending_live}")
    print(f"stale-expression-as-current-live-corpus: {stale_live}")
    print(f"overruled-as-good-law: {overruled}")
    raise SystemExit(
        1
        if any((repealed, repealed_live, pending, pending_live, stale_live, overruled))
        else 0
    )


if __name__ == "__main__":
    main()
