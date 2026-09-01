#!/usr/bin/env python3
"""Recompute document content_hashes after canonical hash changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.pipeline import _content_hash  # noqa: E402


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def load_laws(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source_id = str(record.get("_id") or record.get("name") or "")
        records[source_id] = record
    return records


def law_documents(conn: PgConnection) -> list[tuple[str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_id, content_hash
            FROM documents
            WHERE source_type IN ('act', 'regulation')
            ORDER BY source_id
            """
        )
        return [(str(a), str(b), str(c)) for a, b, c in cur.fetchall()]


def run(
    conn: PgConnection, records: dict[str, dict[str, Any]], *, dry_run: bool = False
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for document_id, source_id, old_hash in law_documents(conn):
        record = records.get(source_id)
        if record is None:
            counts["missing_record"] += 1
            continue
        new_hash = _content_hash(str(record.get("content") or ""))
        if new_hash == old_hash:
            counts["unchanged"] += 1
            continue
        counts["would_update" if dry_run else "updated"] += 1
        if dry_run:
            conn.rollback()
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET content_hash=%s WHERE id=%s",
                    (new_hash, document_id),
                )
            conn.commit()
        except Exception:  # noqa: BLE001 - one bad document must not stop the batch
            conn.rollback()
            counts["failed"] += 1
    return counts


def print_summary(counts: Counter[str], *, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "recompute"
    print(f"{mode} complete.")
    print(f"  unchanged:       {counts.get('unchanged', 0)}")
    print(f"  missing record:  {counts.get('missing_record', 0)}")
    print(f"  would update:    {counts.get('would_update', 0)}")
    print(f"  updated:         {counts.get('updated', 0)}")
    print(f"  failed:          {counts.get('failed', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "laws.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_laws(args.input)
    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        counts = run(conn, records, dry_run=args.dry_run)
        print_summary(counts, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
