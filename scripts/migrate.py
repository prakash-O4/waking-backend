from __future__ import annotations

import os
from pathlib import Path

import psycopg2


MIGRATION = (
    Path(__file__).resolve().parents[1] / "migrations" / "001_bitemporal_schema.sql"
)


def main() -> None:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("SUPABASE_DB_URL not set; skipping Supabase migration")
        return
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
    print("Supabase migration applied")


if __name__ == "__main__":
    main()
