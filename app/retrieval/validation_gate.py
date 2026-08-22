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
            SELECT act_name, case_id, source_type
            FROM chunks
            WHERE id = %(component_uri)s::uuid
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
        "source_kind": row[2],
        "ocr_confidence": None,
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
