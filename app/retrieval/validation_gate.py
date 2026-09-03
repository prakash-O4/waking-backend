from __future__ import annotations

import hashlib
import unicodedata
from datetime import date
from typing import Any

from psycopg2.extensions import connection

from app.retrieval.eligibility_gate import eligible_chunk_ids, is_eligible

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# ponytail: tunable floor against degenerate short-quote matches; revisit with eval data
_MIN_QUOTE_CHARS = 15


def _normalize(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFC", text).translate(_DEVANAGARI_DIGITS).split()
    )


def _claim_supported(quote: str, chunk_text: str) -> bool:
    q = _normalize(quote)
    return len(q) >= _MIN_QUOTE_CHARS and q in _normalize(chunk_text)


_DERIVED_KINDS = {"verified_internal_consolidation", "derived_verified"}


def _citation(conn: connection, evidence_id: str, as_of: date) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.act_name, c.case_id, c.source_type,
                   sp.kind, d.ocr_confidence, sp.source_url
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN source_publication sp ON sp.id = d.source_pub_id
            WHERE c.id = %(evidence_id)s::uuid
            LIMIT 1
            """,
            {"evidence_id": evidence_id},
        )
        row = cur.fetchone()
    if not row:
        return None

    component_uri = _authority_component_uri(conn, evidence_id)
    title_ne = row[0] or row[1] or ""
    title_en = ""
    amendments: list[dict[str, Any]] = []

    if component_uri is not None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT w.title_ne, w.title_en
                FROM component c
                JOIN work w ON w.id = c.work_id
                WHERE c.uri = %(component_uri)s
                LIMIT 1
                """,
                {"component_uri": component_uri},
            )
            comp = cur.fetchone()
        if comp:
            title_ne = comp[0] or title_ne
            title_en = comp[1] or ""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT le.effective_date, sp.kind, sp.ocr_confidence, sp.source_url
                FROM lifecycle_effect le
                LEFT JOIN source_publication sp ON sp.id = le.source_pub_id
                WHERE le.component_uri = %(component_uri)s
                  AND le.effect_type = 'amend'
                  AND le.approval_status = 'approved'
                  AND lower(le.legal_valid_time) <= %(as_of)s
                ORDER BY le.effective_date ASC
                """,
                {"component_uri": component_uri, "as_of": as_of},
            )
            for eff_date, kind, ocr, url in cur.fetchall():
                amendments.append(
                    {
                        "effective_date": eff_date.isoformat() if eff_date else None,
                        "source_kind": kind,
                        "ocr_confidence": ocr,
                        "source_url": url,
                    }
                )

    source_kind = row[3] or row[2]
    return {
        "component_uri": component_uri or evidence_id,
        "work_title_ne": title_ne,
        "work_title_en": title_en,
        "as_of": as_of.isoformat(),
        "source_kind": source_kind,
        "ocr_confidence": row[4],
        "source_url": row[5],
        "derived": source_kind in _DERIVED_KINDS,
        "amendments": amendments,
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
    return component_uri is not None and not is_eligible(conn, component_uri, as_of)


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
        ok = ok and _claim_supported(claim.get("quote", ""), expr[0] if expr else "")
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
