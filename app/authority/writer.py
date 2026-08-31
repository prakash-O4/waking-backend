from __future__ import annotations

from datetime import date
from uuid import UUID

import psycopg2
from psycopg2.extensions import connection

from app.authority.parser import ParsedComponent, ParsedLaw

PHASE0_APPROVER = UUID("00000000-0000-0000-0000-000000000001")


def connect() -> connection:
    import os

    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
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


def propose_lifecycle_commence(
    conn: connection,
    component_uri: str,
    source_pub_id: str,
    *,
    effective_date: date | None,
    commencement_dependency: str | None,
    raw_clause_text: str,
) -> None:
    """Write a pending commencement proposal.

    NEVER sets approval_status to anything but 'pending' — approval is
    exclusively scripts/review_lifecycle.py's dual sign-off path.
    """
    valid_time = f"[{effective_date.isoformat()},)" if effective_date else "empty"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM lifecycle_effect
            WHERE component_uri=%s AND effect_type='commence'
              AND approval_status='pending'
            LIMIT 1
            """,
            (component_uri,),
        )
        if cur.fetchone():
            return
        cur.execute(
            """
            INSERT INTO lifecycle_effect (
                component_uri, effect_type, legal_valid_time, transaction_time,
                effective_date, commencement_dependency, source_pub_id,
                approval_status, raw_clause_text
            ) VALUES (%s, 'commence', %s::tstzrange, tstzrange(now(), NULL),
                      %s, %s, %s, 'pending', %s)
            """,
            (
                component_uri,
                valid_time,
                effective_date,
                commencement_dependency,
                source_pub_id,
                raw_clause_text,
            ),
        )


def propose_lifecycle_amend(
    conn: connection,
    component_uri: str,
    source_pub_id: str,
    *,
    effective_date: date | None,
    amendment_dependency: str | None,
    raw_clause_text: str,
) -> None:
    """Write a pending amendment proposal for one component."""
    valid_time = f"[{effective_date.isoformat()},)" if effective_date else "empty"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM lifecycle_effect
            WHERE component_uri=%s AND effect_type='amend'
              AND approval_status='pending' AND raw_clause_text=%s
            LIMIT 1
            """,
            (component_uri, raw_clause_text),
        )
        if cur.fetchone():
            return
        cur.execute(
            """
            INSERT INTO lifecycle_effect (
                component_uri, effect_type, legal_valid_time, transaction_time,
                effective_date, commencement_dependency, source_pub_id,
                approval_status, raw_clause_text
            ) VALUES (%s, 'amend', %s::tstzrange, tstzrange(now(), NULL),
                      %s, %s, %s, 'pending', %s)
            """,
            (
                component_uri,
                valid_time,
                effective_date,
                amendment_dependency,
                source_pub_id,
                raw_clause_text,
            ),
        )


def propose_lifecycle_repeal(
    conn: connection,
    repealed_work_id: str,
    source_pub_id: str,
    *,
    repealing_work_uri: str,
    raw_clause_text: str,
) -> None:
    """Write pending repeal proposals for every component in a repealed work.

    The repeal date depends on the repealing work's commencement; until that is
    approved/resolved, legal_valid_time stays empty (PS-2).
    """
    dependency: str | None = f"repealing_work_commencement:{repealing_work_uri}"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MIN(effective_date) FROM lifecycle_effect
            WHERE component_uri LIKE %s AND effect_type='commence'
              AND approval_status='approved' AND effective_date IS NOT NULL
            """,
            (repealing_work_uri.rstrip("/") + "/%",),
        )
        date_row = cur.fetchone()
        effective_date = date_row[0] if date_row and date_row[0] else None
        valid_time = f"[{effective_date.isoformat()},)" if effective_date else "empty"
        if effective_date:
            dependency = None

        cur.execute("SELECT uri FROM work WHERE id=%s LIMIT 1", (repealed_work_id,))
        row = cur.fetchone()
        if not row:
            return
        cur.execute(
            "SELECT uri FROM component WHERE uri LIKE %s ORDER BY uri",
            (str(row[0]).rstrip("/") + "/%",),
        )
        component_uris = [str(component_row[0]) for component_row in cur.fetchall()]
        for component_uri in component_uris:
            cur.execute(
                """
                SELECT 1 FROM lifecycle_effect
                WHERE component_uri=%s AND effect_type='repeal'
                  AND approval_status='pending'
                LIMIT 1
                """,
                (component_uri,),
            )
            if cur.fetchone():
                continue
            cur.execute(
                """
                INSERT INTO lifecycle_effect (
                    component_uri, effect_type, legal_valid_time, transaction_time,
                    effective_date, commencement_dependency, source_pub_id,
                    approval_status, raw_clause_text
                ) VALUES (%s, 'repeal', %s::tstzrange, tstzrange(now(), NULL),
                          %s, %s, %s, 'pending', %s)
                """,
                (
                    component_uri,
                    valid_time,
                    effective_date,
                    dependency,
                    source_pub_id,
                    raw_clause_text,
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
