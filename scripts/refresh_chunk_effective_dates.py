#!/usr/bin/env python3
"""Refresh chunks.effective_date_ad from now-approved commence effects.

chunks.effective_date_ad is a cache written once at ingestion time from
whatever commence effects were approved at that moment (see
pipeline.py::_commence_date). Approving a commence effect after ingestion
does not retroactively update it — this script closes that gap.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SELECT_SQL = """
    SELECT count(*)
    FROM chunks c
    JOIN (
        SELECT component_uri, MIN(effective_date) AS min_effective_date
        FROM lifecycle_effect
        WHERE effect_type = 'commence' AND approval_status = 'approved'
        GROUP BY component_uri
    ) sub ON sub.component_uri = c.component_uri
    WHERE c.effective_date_ad IS NULL AND sub.min_effective_date IS NOT NULL
"""

_UPDATE_SQL = """
    UPDATE chunks c
    SET effective_date_ad = sub.min_effective_date
    FROM (
        SELECT component_uri, MIN(effective_date) AS min_effective_date
        FROM lifecycle_effect
        WHERE effect_type = 'commence' AND approval_status = 'approved'
        GROUP BY component_uri
    ) sub
    WHERE c.component_uri = sub.component_uri
      AND c.effective_date_ad IS NULL
      AND sub.min_effective_date IS NOT NULL
"""


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def refresh(conn: PgConnection, *, dry_run: bool) -> int:
    with conn.cursor() as cur:
        if dry_run:
            cur.execute(_SELECT_SQL)
            row = cur.fetchone()
            return int(row[0]) if row else 0
        cur.execute(_UPDATE_SQL)
        return int(cur.rowcount or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        count = refresh(conn, dry_run=args.dry_run)
        if args.dry_run:
            conn.rollback()
            print(f"dry-run: {count} chunks would get effective_date_ad set")
        else:
            conn.commit()
            print(f"refresh complete: {count} chunks updated")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
