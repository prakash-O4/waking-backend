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
from app.authority.writer import upsert_component, upsert_expression  # noqa: E402
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


def _table_count(conn: PgConnection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _table_counts(conn: PgConnection) -> Counter[str]:
    return Counter(
        {
            table: _table_count(conn, table)
            for table in ("component", "expression", "lifecycle_effect")
        }
    )


def _expected_hashes(law: ParsedLaw) -> dict[str, str]:
    expected: dict[str, str] = {}
    for component in law.components:
        if component.uri in expected:
            raise RuntimeError(f"parser emitted duplicate URI: {component.uri}")
        expected[component.uri] = component.text_hash
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


def _missing_component_uris(
    conn: PgConnection, work_id: str, law: ParsedLaw
) -> list[str]:
    stored = _component_uris(conn, work_id)
    return [
        component.uri for component in law.components if component.uri not in stored
    ]


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


def _upsert_missing_components(conn: PgConnection, work_id: str, law: ParsedLaw) -> int:
    missing = set(_missing_component_uris(conn, work_id, law))
    for component in law.components:
        if component.uri in missing:
            upsert_component(conn, work_id, component)
    return len(missing)


def _upsert_missing_expressions(conn: PgConnection, law: ParsedLaw, as_of: date) -> int:
    missing = _missing_expression_count(conn, law, as_of)
    for component in law.components:
        upsert_expression(conn, component, as_of=as_of)
    return missing


def _component_uris(conn: PgConnection, work_id: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT uri FROM component WHERE work_id=%s", (work_id,))
        return {str(row[0]) for row in cur.fetchall()}


def _lifecycle_statuses(conn: PgConnection, component_uri: str) -> Counter[str]:
    statuses: Counter[str] = Counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT approval_status, COUNT(*)
            FROM lifecycle_effect
            WHERE component_uri=%s
            GROUP BY approval_status
            """,
            (component_uri,),
        )
        for status, count in cur.fetchall():
            statuses[str(status)] = int(count)
    return statuses


def _row_count(conn: PgConnection, table: str, column: str, value: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=%s", (value,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _delete_orphan_component(conn: PgConnection, component_uri: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM expression WHERE component_uri=%s", (component_uri,))
        cur.execute(
            "DELETE FROM lifecycle_effect WHERE component_uri=%s", (component_uri,)
        )
        cur.execute("DELETE FROM component WHERE uri=%s", (component_uri,))


def _cleanup_orphan_components(
    conn: PgConnection, work_id: str, law: ParsedLaw, *, dry_run: bool
) -> Counter[str]:
    counts: Counter[str] = Counter()
    expected_uris = {component.uri for component in law.components}
    orphan_uris = _component_uris(conn, work_id) - expected_uris
    for component_uri in orphan_uris:
        statuses = _lifecycle_statuses(conn, component_uri)
        if any(status != "pending" for status in statuses):
            counts["orphan_components_blocked"] += 1
            print(
                f"WARNING {component_uri}: orphan has non-pending lifecycle rows {dict(statuses)}",
                file=sys.stderr,
            )
            continue
        counts["orphan_components"] += 1
        counts["orphan_expressions"] += _row_count(
            conn, "expression", "component_uri", component_uri
        )
        counts["orphan_lifecycle_effects"] += sum(statuses.values())
        if not dry_run:
            _delete_orphan_component(conn, component_uri)
    return counts


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
) -> tuple[str, Counter[str]]:
    record = records.get(source_id)
    if record is None:
        return "missing_record", Counter()

    content = str(record.get("content") or "")
    if _content_hash(content) != content_hash:
        return "hash_mismatch", Counter()

    law = parse_law(record)
    work_id = work_id_for_document(conn, document_id)
    if not work_id:
        return "missing_work_id", Counter()

    stored_uri = work_uri(conn, work_id)
    if stored_uri != law.uri:
        print(
            f"WARNING {source_id}: law.uri != work.uri ({law.uri!r} != {stored_uri!r})",
            file=sys.stderr,
        )
        return "uri_mismatch", Counter()

    stale_ids = _stale_expression_ids(conn, law, as_of)
    counts = _cleanup_orphan_components(conn, work_id, law, dry_run=dry_run)
    counts["stale_expressions"] += len(stale_ids)
    counts["missing_components"] += len(_missing_component_uris(conn, work_id, law))
    counts["missing_expressions"] += _missing_expression_count(conn, law, as_of)
    counts["source_sha_updates"] += _fix_source_sha(
        conn, work_id, content, law, dry_run=dry_run
    )
    if not dry_run:
        _upsert_missing_components(conn, work_id, law)
        _upsert_missing_expressions(conn, law, as_of)
        _delete_expressions(conn, stale_ids)
    return "processed", counts


def run_cleanup(
    conn: PgConnection,
    records: dict[str, dict[str, Any]],
    *,
    as_of: date | None = None,
    dry_run: bool = False,
) -> Summary:
    summary = Summary()
    as_of = as_of or date.today()
    for table, count in _table_counts(conn).items():
        summary.counts[f"{table}_before"] = count

    for document_id, source_id, _source_type, content_hash in law_documents(conn):
        try:
            status, counts = cleanup_document(
                conn,
                document_id=document_id,
                source_id=source_id,
                content_hash=content_hash,
                records=records,
                as_of=as_of,
                dry_run=dry_run,
            )
            summary.counts[status] += 1
            summary.counts.update(counts)
            conn.rollback() if dry_run else conn.commit()
        except Exception as exc:  # noqa: BLE001 - keep batch moving
            conn.rollback()
            summary.counts["write_exception"] += 1
            if len(summary.failed_source_ids) < 10:
                summary.failed_source_ids.append(source_id)
            print(f"WARNING {source_id}: cleanup failed: {exc}", file=sys.stderr)

    for table, count in _table_counts(conn).items():
        summary.counts[f"{table}_after"] = count
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
    print(f"  components before:         {counts.get('component_before', 0)}")
    print(f"  expressions before:        {counts.get('expression_before', 0)}")
    print(f"  lifecycle rows before:     {counts.get('lifecycle_effect_before', 0)}")
    print(f"  stale expressions found:   {counts.get('stale_expressions', 0)}")
    print(f"  orphan components found:   {counts.get('orphan_components', 0)}")
    print(f"  orphan expressions found:  {counts.get('orphan_expressions', 0)}")
    print(f"  orphan lifecycle found:    {counts.get('orphan_lifecycle_effects', 0)}")
    print(f"  orphan components blocked: {counts.get('orphan_components_blocked', 0)}")
    print(f"  missing components found:  {counts.get('missing_components', 0)}")
    print(f"  missing expressions found: {counts.get('missing_expressions', 0)}")
    print(f"  source sha rows updated:   {counts.get('source_sha_updates', 0)}")
    print(f"  components after:          {counts.get('component_after', 0)}")
    print(f"  expressions after:         {counts.get('expression_after', 0)}")
    print(f"  lifecycle rows after:      {counts.get('lifecycle_effect_after', 0)}")
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
