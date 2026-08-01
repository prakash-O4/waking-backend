from __future__ import annotations

import os

import psycopg2
import pytest


TABLES = {"work", "component", "source_publication", "lifecycle_effect", "expression"}


@pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="SUPABASE_DB_URL not set")
def test_bitemporal_tables_exist() -> None:
    with psycopg2.connect(os.environ["SUPABASE_DB_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
            """,
            (list(TABLES),),
        )
        assert {row[0] for row in cur.fetchall()} == TABLES
