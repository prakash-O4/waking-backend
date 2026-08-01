from __future__ import annotations

from datetime import date

from psycopg2.extensions import connection


def is_eligible(conn: connection, component_uri: str, as_of: date) -> bool:
    """Calls is_eligible() SQL function."""
    with conn.cursor() as cur:
        cur.execute("SELECT is_eligible(%s, %s)", (component_uri, as_of))
        row = cur.fetchone()
        return bool(row and row[0])
