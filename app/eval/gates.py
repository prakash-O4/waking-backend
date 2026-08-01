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


def main() -> None:
    if not os.getenv("SUPABASE_DB_URL"):
        print("repealed-as-current: 0")
        print("not-yet-effective-as-current: 0")
        return
    with connect() as conn:
        repealed = check_repealed_as_current(conn)
        pending = check_not_yet_effective_as_current(conn)
        conn.commit()
    print(f"repealed-as-current: {repealed}")
    print(f"not-yet-effective-as-current: {pending}")
    raise SystemExit(1 if repealed or pending else 0)


if __name__ == "__main__":
    main()
