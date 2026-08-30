"""
Backfill enabling-power links for already-ingested नियमावली/नियमहरू documents.

Connects to Postgres, finds regulation documents, and runs the same extractor
used at ingestion time. Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import psycopg2
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.enabling_extractor import extract_enabling_clause  # noqa: E402


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def _regulation_work_ids(conn: PgConnection) -> list[tuple[str, str]]:
    """Return (work_id, raw_content) pairs for regulation documents."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT c.work_id, d.raw_content
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            WHERE d.source_type = 'regulation'
              AND c.work_id IS NOT NULL
            """
        )
        return [(str(row[0]), str(row[1])) for row in cur.fetchall()]


def _count_statuses(conn: PgConnection) -> Counter[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT resolution_status, COUNT(*)
            FROM work_relations
            GROUP BY resolution_status
            """
        )
        return Counter({row[0]: row[1] for row in cur.fetchall()})


def main() -> None:
    url = _db_url()
    conn = psycopg2.connect(url)
    conn.autocommit = False

    try:
        pairs = _regulation_work_ids(conn)
        print(f"Found {len(pairs)} already-ingested regulation documents")

        for work_id, raw_content in pairs:
            extract_enabling_clause(
                content=raw_content,
                work_id=work_id,
                conn=conn,
                source_id=f"backfill:{work_id}",
            )

        conn.commit()
        counts = _count_statuses(conn)
        print("Backfill complete.")
        print(f"  resolved:                {counts.get('auto_extracted', 0)}")
        print(f"  parent_not_in_corpus:    {counts.get('parent_not_in_corpus', 0)}")
        print(f"  no_enabling_clause:      {counts.get('no_enabling_clause', 0)}")
        print(f"  human_verified:          {counts.get('human_verified', 0)}")
        print(f"  false_positive:          {counts.get('false_positive', 0)}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
