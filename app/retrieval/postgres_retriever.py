from __future__ import annotations

from datetime import date
from typing import Any

from psycopg2.extensions import connection

from app.retrieval.eligibility_gate import is_eligible


def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    """ILIKE fallback retriever, guarded by Postgres eligibility."""
    tokens = [t for t in query.split() if t][:5]
    if not tokens:
        return []

    where = " AND ".join(["e.text_ne ILIKE %s"] * len(tokens))
    params = [f"%{t}%" for t in tokens]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.uri, e.text_ne, e.text_hash, w.title_ne
            FROM expression e
            JOIN component c ON c.uri = e.component_uri
            JOIN work w ON w.id = c.work_id
            WHERE {where} AND e.as_of <= %s
            ORDER BY e.as_of DESC
            LIMIT 50
            """,
            (*params, as_of),
        )
        rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for component_uri, text_ne, text_hash, title_ne in rows:
        if component_uri in seen or not is_eligible(conn, component_uri, as_of):
            continue
        seen.add(component_uri)
        results.append(
            {
                "component_uri": component_uri,
                "text_ne": text_ne,
                "text_hash": text_hash,
                "score": 1.0,
                "work_title_ne": title_ne,
            }
        )
        if len(results) >= k:
            break
    return results
