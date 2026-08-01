from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from psycopg2.extensions import connection

from app.retrieval.eligibility_gate import is_eligible


def _citation(
    conn: connection, component_uri: str, as_of: date
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT w.title_ne, w.title_en, sp.kind, sp.ocr_confidence
            FROM component c
            JOIN work w ON w.id = c.work_id
            LEFT JOIN source_publication sp ON sp.work_id = w.id
            WHERE c.uri = %s
            ORDER BY sp.ingested_at DESC NULLS LAST
            LIMIT 1
            """,
            (component_uri,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "component_uri": component_uri,
        "work_title_ne": row[0],
        "work_title_en": row[1],
        "as_of": as_of.isoformat(),
        "source_kind": row[2],
        "ocr_confidence": row[3],
    }


def _expression(
    conn: connection, component_uri: str, as_of: date
) -> tuple[str, str] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text_ne, text_hash
            FROM expression
            WHERE component_uri = %s AND as_of = %s
            LIMIT 1
            """,
            (component_uri, as_of),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                """
                SELECT text_ne, text_hash
                FROM expression
                WHERE component_uri = %s AND as_of <= %s
                ORDER BY as_of DESC
                LIMIT 1
                """,
                (component_uri, as_of),
            )
            row = cur.fetchone()
    return (row[0], row[1]) if row else None


def validate_and_render(
    claims: list[dict[str, str]], as_of: date, conn: connection
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for claim in claims:
        component_uri = claim.get("evidence_id", "")
        expr = _expression(conn, component_uri, as_of)
        ok = False
        if expr:
            ok = hashlib.sha256(expr[0].encode("utf-8")).hexdigest() == expr[1]
        ok = ok and is_eligible(conn, component_uri, as_of)
        citation = _citation(conn, component_uri, as_of) if ok else None
        rendered.append(
            {
                "claim": claim.get("claim", ""),
                "evidence_id": component_uri,
                "abstained": citation is None,
                "citation": citation,
            }
        )
    return rendered
