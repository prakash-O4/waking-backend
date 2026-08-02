from __future__ import annotations

import os
from pathlib import Path

import psycopg2


MIGRATIONS = [
    Path(__file__).resolve().parents[1] / "migrations" / "001_bitemporal_schema.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "002_gate_suspend_fix.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "003_bs_ad_calendar.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "004_precedent_schema.sql",
]


def main() -> None:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("SUPABASE_DB_URL not set; skipping Supabase migration")
        return
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        for migration in MIGRATIONS:
            cur.execute(migration.read_text())
    print("Supabase migrations applied")


if __name__ == "__main__":
    main()
