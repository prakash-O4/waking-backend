#!/usr/bin/env python3
"""Review pending documents for retrieval eligibility."""

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


def list_pending(conn: PgConnection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_type, source_id, ingested_at, redaction_failed
            FROM documents
            WHERE ingestion_status='pending'
            ORDER BY ingested_at, id
            """
        )
        rows = cur.fetchall()
    for row in rows:
        print(
            f"{row[0]}  {row[1]}  {row[2]}  ingested={row[3]} "
            f"redaction_failed={row[4]}"
        )
    if not rows:
        print("no pending documents")


def _approve_one(conn: PgConnection, document_id: str, by: str) -> tuple[bool, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ingestion_status, approved_by, second_approved_by, redaction_failed
            FROM documents WHERE id=%s FOR UPDATE
            """,
            (document_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, "not found"
        status, approver1, approver2, redaction_failed = row
        if redaction_failed:
            return False, "redaction verification failed, cannot approve"
        if status != "pending":
            return False, f"already {status}"
        if approver1 is None:
            cur.execute(
                "UPDATE documents SET approved_by=%s WHERE id=%s",
                (by, document_id),
            )
            return (
                True,
                "recorded as first approver — a second, different person must approve before this is retrievable.",
            )
        if str(approver1) == by:
            return False, "same person cannot be both approvers"
        if approver2 is None:
            cur.execute(
                """
                UPDATE documents
                SET second_approved_by=%s, ingestion_status='approved'
                WHERE id=%s
                """,
                (by, document_id),
            )
            return True, "approved"
    return False, "already has two approvers"


def approve_one(conn: PgConnection, document_id: str, by: str) -> bool:
    ok, message = _approve_one(conn, document_id, by)
    conn.commit() if ok else conn.rollback()
    print(message)
    return ok


def _reject_one(conn: PgConnection, document_id: str, by: str) -> tuple[bool, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ingestion_status FROM documents WHERE id=%s FOR UPDATE",
            (document_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, "not found"
        if row[0] != "pending":
            return False, f"already {row[0]}"
        cur.execute(
            """
            UPDATE documents
            SET ingestion_status='rejected', approved_by=%s
            WHERE id=%s
            """,
            (by, document_id),
        )
    return True, "rejected"


def reject_one(conn: PgConnection, document_id: str, by: str) -> bool:
    ok, message = _reject_one(conn, document_id, by)
    conn.commit() if ok else conn.rollback()
    print(message)
    return ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--approve")
    group.add_argument("--reject")
    parser.add_argument("--by", help="reviewer name")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (args.approve or args.reject) and not args.by:
        raise SystemExit("--by is required")

    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        if args.list:
            list_pending(conn)
        elif args.approve:
            raise SystemExit(0 if approve_one(conn, args.approve, args.by) else 1)
        elif args.reject:
            raise SystemExit(0 if reject_one(conn, args.reject, args.by) else 1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
