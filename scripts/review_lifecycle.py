#!/usr/bin/env python3
"""Review pending lifecycle_effect proposals."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import errors
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def _work_uri(component_uri: str) -> str:
    for marker in ("/dafa/", "/dhara/", "/parichheda/", "/full/"):
        if marker in component_uri:
            return component_uri.split(marker, 1)[0]
    return component_uri.rsplit("/", 2)[0]


def list_pending(
    conn: PgConnection, work_uri: str | None = None, effect_type: str | None = None
) -> None:
    where = "approval_status='pending'"
    params: tuple[Any, ...] = ()
    if effect_type:
        where += " AND effect_type=%s"
        params = (effect_type,)
    if work_uri:
        where += " AND component_uri LIKE %s"
        params += (work_uri.rstrip("/") + "/%",)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, component_uri, effective_date, commencement_dependency,
                   COALESCE(raw_clause_text, ''), effect_type
            FROM lifecycle_effect
            WHERE {where}
            ORDER BY effect_type, component_uri, raw_clause_text, id
            """,
            params,
        )
        rows = cur.fetchall()

    current: tuple[str, str] | None = None
    for row in rows:
        row_work = _work_uri(str(row[1]))
        key = (row_work, str(row[4]))
        if key != current:
            current = key
            clause = str(row[4]) or "<no_commencement_clause>"
            print(f"\n{row_work}\n  clause: {clause}")
        print(f"  {row[0]}  {row[5]}  {row[1]}  effective={row[2]} dependency={row[3]}")
    if not rows:
        print("no pending lifecycle proposals")


def _approve_one(conn: PgConnection, proposal_id: str, by: str) -> tuple[bool, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT approval_status, approved_by_1, approved_by_2
            FROM lifecycle_effect WHERE id=%s FOR UPDATE
            """,
            (proposal_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, "not found"
        status, approver1, approver2 = row
        if status != "pending":
            return False, f"already {status}"
        if approver1 is None:
            cur.execute(
                "UPDATE lifecycle_effect SET approved_by_1=%s WHERE id=%s",
                (by, proposal_id),
            )
            return (
                True,
                "recorded as first approver — a second, different person must approve before this takes effect.",
            )
        if str(approver1) == by:
            return False, "same person cannot be both approvers"
        if approver2 is None:
            cur.execute(
                """
                UPDATE lifecycle_effect
                SET approved_by_2=%s, approval_status='approved'
                WHERE id=%s
                """,
                (by, proposal_id),
            )
            return True, "approved"
    return False, "already has two approvers"


def approve_one(conn: PgConnection, proposal_id: str, by: str) -> bool:
    try:
        ok, message = _approve_one(conn, proposal_id, by)
        conn.commit() if ok else conn.rollback()
    except errors.ExclusionViolation:
        conn.rollback()
        print("approval would overlap an already-approved lifecycle effect; refused")
        return False
    print(message)
    return ok


def _reject_one(conn: PgConnection, proposal_id: str, by: str) -> tuple[bool, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT approval_status FROM lifecycle_effect WHERE id=%s FOR UPDATE",
            (proposal_id,),
        )
        row = cur.fetchone()
        if not row:
            return False, "not found"
        if row[0] != "pending":
            return False, f"already {row[0]}"
        # Reject path reuses approved_by_1 as single-decision attribution.
        cur.execute(
            """
            UPDATE lifecycle_effect
            SET approval_status='rejected', approved_by_1=%s
            WHERE id=%s
            """,
            (by, proposal_id),
        )
    return True, "rejected"


def reject_one(conn: PgConnection, proposal_id: str, by: str) -> bool:
    ok, message = _reject_one(conn, proposal_id, by)
    conn.commit() if ok else conn.rollback()
    print(message)
    return ok


def _pending_for_work(
    conn: PgConnection, work_uri: str, effect_type: str | None = None
) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM lifecycle_effect
            WHERE component_uri LIKE %s AND (%s IS NULL OR effect_type=%s)
              AND approval_status='pending'
            ORDER BY component_uri, id
            """,
            (work_uri.rstrip("/") + "/%", effect_type, effect_type),
        )
        return [str(row[0]) for row in cur.fetchall()]


def approve_work(
    conn: PgConnection, work_uri: str, by: str, effect_type: str | None = None
) -> bool:
    ok_all = True
    for proposal_id in _pending_for_work(conn, work_uri, effect_type):
        print(f"{proposal_id}: ", end="")
        ok_all = approve_one(conn, proposal_id, by) and ok_all
    return ok_all


def reject_work(
    conn: PgConnection, work_uri: str, by: str, effect_type: str | None = None
) -> bool:
    ok_all = True
    for proposal_id in _pending_for_work(conn, work_uri, effect_type):
        print(f"{proposal_id}: ", end="")
        ok_all = reject_one(conn, proposal_id, by) and ok_all
    return ok_all


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--list-work")
    group.add_argument("--approve")
    group.add_argument("--reject")
    group.add_argument("--approve-work")
    group.add_argument("--reject-work")
    parser.add_argument("--by", help="reviewer UUID")
    parser.add_argument("--effect-type", choices=("commence", "repeal"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        args.approve or args.reject or args.approve_work or args.reject_work
    ) and not args.by:
        raise SystemExit("--by is required")

    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        if args.list:
            list_pending(conn, effect_type=args.effect_type)
        elif args.list_work:
            list_pending(conn, args.list_work, args.effect_type)
        elif args.approve:
            raise SystemExit(0 if approve_one(conn, args.approve, args.by) else 1)
        elif args.reject:
            raise SystemExit(0 if reject_one(conn, args.reject, args.by) else 1)
        elif args.approve_work:
            raise SystemExit(
                0
                if approve_work(conn, args.approve_work, args.by, args.effect_type)
                else 1
            )
        elif args.reject_work:
            raise SystemExit(
                0
                if reject_work(conn, args.reject_work, args.by, args.effect_type)
                else 1
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
