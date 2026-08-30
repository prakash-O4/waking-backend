#!/usr/bin/env python3
"""Delete expression rows created by the old loose-header parser."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PgConnection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.authority.parser import ParsedLaw, parse_law  # noqa: E402
from app.authority.writer import upsert_expression  # noqa: E402
from app.ingestion.pipeline import _content_hash  # noqa: E402
from scripts.backfill_authority_layer import (  # noqa: E402
    law_documents,
    load_laws,
    work_id_for_document,
    work_uri,
)


@dataclass
class Summary:
    counts: Counter[str] = field(default_factory=Counter)
    failed_source_ids: list[str] = field(default_factory=list)


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set")
    return url


def _raw_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _expression_count(conn: PgConnection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM expression")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _expected_hashes(law: ParsedLaw) -> dict[str, str]:
    expected: dict[str, str] = {}
    for component in law.components:
        expected.setdefault(component.uri, component.text_hash)
    return expected


def _stale_expression_ids(conn: PgConnection, law: ParsedLaw, as_of: date) -> list[str]:
    stale: list[str] = []
    with conn.cursor() as cur:
        for component_uri, text_hash in _expected_hashes(law).items():
            cur.execute(
                """
                SELECT id
                FROM expression
                WHERE component_uri=%s AND as_of=%s AND text_hash<>%s
                """,
                (component_uri, as_of, text_hash),
            )
            stale.extend(str(row[0]) for row in cur.fetchall())
    return stale


def _delete_expressions(conn: PgConnection, ids: list[str]) -> None:
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM expression WHERE id = ANY(%s::uuid[])", (ids,))


def _missing_expression_count(conn: PgConnection, law: ParsedLaw, as_of: date) -> int:
    missing = 0
    with conn.cursor() as cur:
        for component_uri, text_hash in _expected_hashes(law).items():
            cur.execute(
                """
                SELECT 1 FROM expression
                WHERE component_uri=%s AND as_of=%s AND text_hash=%s LIMIT 1
                """,
                (component_uri, as_of, text_hash),
            )
            if not cur.fetchone():
                missing += 1
    return missing


def _upsert_missing_expressions(conn: PgConnection, law: ParsedLaw, as_of: date) -> int:
    missing = _missing_expression_count(conn, law, as_of)
    seen: set[str] = set()
    for component in law.components:
        if component.uri in seen:
            continue
        seen.add(component.uri)
        upsert_expression(conn, component, as_of=as_of)
    return missing


def _fix_source_sha(
    conn: PgConnection, work_id: str, content: str, law: ParsedLaw, *, dry_run: bool
) -> int:
    raw = _raw_sha256(content)
    if raw == law.source_sha256:
        return 0
    with conn.cursor() as cur:
        if dry_run:
            cur.execute(
                "SELECT COUNT(*) FROM source_publication WHERE work_id=%s AND sha256=%s",
                (work_id, raw),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        cur.execute(
            """
            UPDATE source_publication
            SET sha256=%s
            WHERE work_id=%s AND sha256=%s
            """,
            (law.source_sha256, work_id, raw),
        )
        return int(cur.rowcount or 0)


def cleanup_document(
    conn: PgConnection,
    *,
    document_id: str,
    source_id: str,
    content_hash: str,
    records: dict[str, dict[str, Any]],
    as_of: date,
    dry_run: bool,
) -> tuple[str, int, int, int]:
    record = records.get(source_id)
    if record is None:
        return "missing_record", 0, 0, 0

    content = str(record.get("content") or "")
    if _content_hash(content) != content_hash:
        return "hash_mismatch", 0, 0, 0

    law = parse_law(record)
    work_id = work_id_for_document(conn, document_id)
    if not work_id:
        return "missing_work_id", 0, 0, 0

    stored_uri = work_uri(conn, work_id)
    if stored_uri != law.uri:
        print(
            f"WARNING {source_id}: law.uri != work.uri ({law.uri!r} != {stored_uri!r})",
            file=sys.stderr,
        )
        return "uri_mismatch", 0, 0, 0

    stale_ids = _stale_expression_ids(conn, law, as_of)
    missing = _missing_expression_count(conn, law, as_of)
    sha_updates = _fix_source_sha(conn, work_id, content, law, dry_run=dry_run)
    if not dry_run:
        _upsert_missing_expressions(conn, law, as_of)
        _delete_expressions(conn, stale_ids)
    return "processed", len(stale_ids), sha_updates, missing


def run_cleanup(
    conn: PgConnection,
    records: dict[str, dict[str, Any]],
    *,
    as_of: date | None = None,
    dry_run: bool = False,
) -> Summary:
    summary = Summary()
    as_of = as_of or date.today()
    summary.counts["expressions_before"] = _expression_count(conn)

    for document_id, source_id, _source_type, content_hash in law_documents(conn):
        try:
            status, stale, sha_updates, missing = cleanup_document(
                conn,
                document_id=document_id,
                source_id=source_id,
                content_hash=content_hash,
                records=records,
                as_of=as_of,
                dry_run=dry_run,
            )
            summary.counts[status] += 1
            summary.counts["stale_expressions"] += stale
            summary.counts["source_sha_updates"] += sha_updates
            summary.counts["missing_expressions"] += missing
            conn.rollback() if dry_run else conn.commit()
        except Exception as exc:  # noqa: BLE001 - keep batch moving
            conn.rollback()
            summary.counts["write_exception"] += 1
            if len(summary.failed_source_ids) < 10:
                summary.failed_source_ids.append(source_id)
            print(f"WARNING {source_id}: cleanup failed: {exc}", file=sys.stderr)

    summary.counts["expressions_after"] = _expression_count(conn)
    return summary


def print_summary(summary: Summary, *, dry_run: bool) -> None:
    counts = summary.counts
    print(("dry-run" if dry_run else "cleanup") + " complete.")
    print(f"  processed:                 {counts.get('processed', 0)}")
    print(f"  skipped missing record:    {counts.get('missing_record', 0)}")
    print(f"  skipped hash mismatch:     {counts.get('hash_mismatch', 0)}")
    print(f"  skipped missing work_id:   {counts.get('missing_work_id', 0)}")
    print(f"  skipped uri mismatch:      {counts.get('uri_mismatch', 0)}")
    print(f"  skipped write exception:   {counts.get('write_exception', 0)}")
    print(f"  expressions before:        {counts.get('expressions_before', 0)}")
    print(f"  stale expressions found:   {counts.get('stale_expressions', 0)}")
    print(f"  missing expressions found: {counts.get('missing_expressions', 0)}")
    print(f"  source sha rows updated:   {counts.get('source_sha_updates', 0)}")
    print(f"  expressions after:         {counts.get('expressions_after', 0)}")
    if summary.failed_source_ids:
        print(f"  write exception samples:   {', '.join(summary.failed_source_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "laws.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        summary = run_cleanup(
            conn, load_laws(args.input), as_of=args.as_of, dry_run=args.dry_run
        )
        print_summary(summary, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
