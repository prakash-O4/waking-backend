from __future__ import annotations

from datetime import date
from uuid import UUID

import psycopg2
from psycopg2.extensions import connection

from app.authority.parser import ParsedComponent, ParsedLaw

PHASE0_APPROVER = UUID("00000000-0000-0000-0000-000000000001")


def connect() -> connection:
    import os

    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL is not set")
    return psycopg2.connect(url)


def upsert_work(conn: connection, law: ParsedLaw) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO work (uri, work_type, title_ne, title_en)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uri) DO UPDATE
            SET title_ne = EXCLUDED.title_ne, title_en = EXCLUDED.title_en
            RETURNING id
            """,
            (law.uri, law.work_type.value, law.title_ne, law.title_en),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("work insert returned no id")
        return str(row[0])


def upsert_source(
    conn: connection, work_id: str, law: ParsedLaw, source_url: str | None
) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM source_publication WHERE work_id=%s AND sha256=%s LIMIT 1",
            (work_id, law.source_sha256),
        )
        row = cur.fetchone()
        if row:
            return str(row[0])
        cur.execute(
            """
            INSERT INTO source_publication (work_id, kind, source_url, sha256, ocr_confidence)
            VALUES (%s, 'official_copy_unverified', %s, %s, NULL)
            RETURNING id
            """,
            (work_id, source_url, law.source_sha256),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("source insert returned no id")
        return str(row[0])


def upsert_component(
    conn: connection, work_id: str, component: ParsedComponent
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO component (work_id, uri, component_type, number)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uri) DO NOTHING
            """,
            (work_id, component.uri, component.component_type, component.number),
        )


def insert_commence(
    conn: connection, component_uri: str, source_pub_id: str, effective_date: date
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM lifecycle_effect
            WHERE component_uri=%s AND effect_type='commence' AND approval_status='approved'
            LIMIT 1
            """,
            (component_uri,),
        )
        if cur.fetchone():
            return
        # Phase 0 stub: lifecycle writes are pre-approved by a fixed service UUID;
        # real dual human approval lands in Phase A.
        cur.execute(
            """
            INSERT INTO lifecycle_effect (
                component_uri, effect_type, legal_valid_time, transaction_time,
                effective_date, commencement_dependency, source_pub_id,
                approval_status, approved_by_1, approved_by_2
            ) VALUES (%s, 'commence', %s::tstzrange, '[now(),)'::tstzrange,
                      %s, NULL, %s, 'approved', %s, %s)
            """,
            (
                component_uri,
                f"[{effective_date.isoformat()},)",
                effective_date,
                source_pub_id,
                str(PHASE0_APPROVER),
                str(PHASE0_APPROVER),
            ),
        )


def upsert_expression(
    conn: connection, component: ParsedComponent, as_of: date
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM expression WHERE component_uri=%s AND as_of=%s AND text_hash=%s LIMIT 1
            """,
            (component.uri, as_of, component.text_hash),
        )
        if cur.fetchone():
            return
        cur.execute(
            """
            INSERT INTO expression (component_uri, as_of, text_ne, text_hash, is_derived)
            VALUES (%s, %s, %s, %s, TRUE)
            """,
            (component.uri, as_of, component.text_ne, component.text_hash),
        )
