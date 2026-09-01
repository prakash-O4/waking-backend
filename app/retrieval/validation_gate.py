from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from psycopg2.extensions import connection

from app.retrieval.eligibility_gate import eligible_chunk_ids


def _citation(
    conn: connection, component_uri: str, as_of: date
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.act_name, c.case_id, c.source_type,
                   sp.kind, d.ocr_confidence
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN source_publication sp ON sp.id = d.source_pub_id
            WHERE c.id = %(component_uri)s::uuid
            LIMIT 1
            """,
            {"component_uri": component_uri},
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "component_uri": component_uri,
        "work_title_ne": row[0] or row[1] or "",
        "work_title_en": "",
        "as_of": as_of.isoformat(),
        "source_kind": row[3] or row[2],
        "ocr_confidence": row[4],
    }


def _expression(
    conn: connection, component_uri: str, as_of: date
) -> tuple[str, str] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_text, span_sha256
            FROM chunks
            WHERE id = %(component_uri)s::uuid
            """,
            {"component_uri": component_uri},
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


def _authority_component_uri(conn: connection, evidence_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT component_uri FROM chunks WHERE id = %(evidence_id)s::uuid",
            {"evidence_id": evidence_id},
        )
        row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def _terminated_before(conn: connection, evidence_id: str, as_of: date) -> bool:
    component_uri = _authority_component_uri(conn, evidence_id)
    if component_uri is None:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM lifecycle_effect
            WHERE component_uri = %(component_uri)s
              AND approval_status = 'approved'
              AND effect_type IN ('repeal', 'expiry', 'suspend')
              AND lower(legal_valid_time) <= %(as_of)s::timestamptz
            LIMIT 1
            """,
            {"component_uri": component_uri, "as_of": as_of},
        )
        return cur.fetchone() is not None


def validate_and_render(
    claims: list[dict[str, str]], as_of: date, conn: connection
) -> list[dict[str, Any]]:
    eligible = eligible_chunk_ids(conn, as_of)
    rendered: list[dict[str, Any]] = []
    for claim in claims:
        component_uri = claim.get("evidence_id", "")
        expr = _expression(conn, component_uri, as_of)
        ok = False
        if expr:
            ok = hashlib.sha256(expr[0].encode("utf-8")).hexdigest() == expr[1]
        ok = ok and component_uri in eligible
        ok = ok and not _terminated_before(conn, component_uri, as_of)
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
