from __future__ import annotations

from datetime import date

from psycopg2.extensions import connection


def is_eligible(conn: connection, component_uri: str, as_of: date) -> bool:
    """Compatibility for legacy eval gates; retrieval uses eligible_chunk_ids."""
    with conn.cursor() as cur:
        cur.execute("SELECT is_eligible(%s, %s)", (component_uri, as_of))
        row = cur.fetchone()
        return bool(row and row[0])


def eligible_chunk_ids(conn: connection, as_of: date) -> set[str]:
    """Return chunk UUIDs eligible for retrieval at as_of."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id::text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.ingestion_status = 'approved'
              AND c.source_type <> 'nkp_case'
              AND c.effective_date_ad IS NOT NULL
              AND c.effective_date_ad <= %(as_of)s
            """,
            {"as_of": as_of},
        )
        return {row[0] for row in cur.fetchall()}
