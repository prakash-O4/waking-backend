#!/usr/bin/env python3
"""Backfill laws.jsonl source_publication rows to derived_verified."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def count_unfixed(conn: PgConnection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM source_publication WHERE kind='official_copy_unverified'"
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def run(conn: PgConnection, *, dry_run: bool = False) -> int:
    count = count_unfixed(conn)
    if dry_run or count == 0:
        conn.rollback()
        return count
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE source_publication
            SET kind='derived_verified'
            WHERE kind='official_copy_unverified'
            """
        )
    conn.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        count = run(conn, dry_run=args.dry_run)
        action = "would update" if args.dry_run else "updated"
        print(f"{action}: {count} source_publication rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
