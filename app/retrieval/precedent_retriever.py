from __future__ import annotations

from datetime import date
from typing import Any

from psycopg2.extensions import connection


def retrieve_precedent(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
    """
    ILIKE search over precedent_holding.holding_text, filtered by is_good_law().
    Returns [{holding_id, holding_text, case_uri, case_title, decided_date}].
    Returns empty list when precedent tables are empty (no corpus yet).
    """
    tokens = query.split()[:5]
    if not tokens:
        return []
    where = " AND ".join(["ph.holding_text ILIKE %s"] * len(tokens))
    params = [f"%{token}%" for token in tokens]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ph.id, ph.holding_text, p.uri, p.title, p.decided_date
            FROM precedent_holding ph
            JOIN precedent p ON p.id = ph.precedent_id
            WHERE {where}
            LIMIT 50
            """,
            params,
        )
        rows = cur.fetchall()
    results: list[dict[str, Any]] = []
    for holding_id, holding_text, case_uri, case_title, decided_date in rows:
        with conn.cursor() as cur:
            cur.execute("SELECT is_good_law(%s, %s)", (holding_id, as_of))
            row = cur.fetchone()
        if row and row[0]:
            results.append(
                {
                    "holding_id": str(holding_id),
                    "holding_text": holding_text,
                    "case_uri": case_uri,
                    "case_title": case_title,
                    "decided_date": decided_date,
                }
            )
        if len(results) >= k:
            break
    return results
