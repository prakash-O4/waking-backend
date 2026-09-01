#!/usr/bin/env python3
"""Backfill authority-layer rows for already-ingested laws."""

from __future__ import annotations

import argparse
import json
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
from app.authority.writer import (  # noqa: E402
    upsert_component,
    upsert_expression,
    upsert_source,
)
from app.ingestion.commencement_extractor import (  # noqa: E402
    classify_commencement,
    extract_commencement_proposals,
)
from app.ingestion.pipeline import _content_hash  # noqa: E402

_ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")


@dataclass
class Summary:
    counts: Counter[str] = field(default_factory=Counter)
    proposal_counts: Counter[str] = field(default_factory=Counter)
    failed_source_ids: list[str] = field(default_factory=list)


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


def law_documents(conn: PgConnection) -> list[tuple[str, str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_id, source_type, content_hash
            FROM documents
            WHERE source_type IN ('act', 'regulation')
            ORDER BY source_id
            """
        )
        return [(str(a), str(b), str(c), str(d)) for a, b, c, d in cur.fetchall()]


def work_id_for_document(conn: PgConnection, document_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT work_id FROM chunks WHERE document_id=%s LIMIT 1",
            (document_id,),
        )
        row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def work_uri(conn: PgConnection, work_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT uri FROM work WHERE id=%s", (work_id,))
        row = cur.fetchone()
    return str(row[0]) if row else None


def _proposal_bucket(law: ParsedLaw, content: str) -> tuple[str, int]:
    proposal = classify_commencement(law, content)
    if proposal.commencement_dependency:
        key = proposal.commencement_dependency
    elif "प्रमाणीकरण भएको" in proposal.raw_clause_text:
        key = "relative-delay-resolved"
    else:
        key = "immediate"
    rows = 1 if key == "no_commencement_clause" else len(law.components)
    return key, rows


def _table_counts(conn: PgConnection) -> Counter[str]:
    counts: Counter[str] = Counter()
    for table in ("component", "source_publication", "expression", "lifecycle_effect"):
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            counts[table] = int(row[0]) if row else 0
    return counts


def _bucket_from_row(dependency: str | None, raw_clause_text: str) -> str:
    if dependency:
        return dependency
    if "प्रमाणीकरण भएको" in raw_clause_text:
        return "relative-delay-resolved"
    return "immediate"


def _section_values(number: str | None) -> tuple[str, str]:
    value = str(number or "")
    return value, value.translate(_ASCII_TO_DEVA)


def _update_authority_links(
    conn: PgConnection, *, document_id: str, source_pub_id: str, law: ParsedLaw
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET source_pub_id=%s WHERE id=%s",
            (source_pub_id, document_id),
        )
    linked = 0
    for component in law.components:
        if component.component_type != "dafa" or not component.number:
            continue
        ascii_number, deva_number = _section_values(component.number)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chunks
                SET component_uri=%s
                WHERE document_id=%s
                  AND (section_number IN (%s, %s) OR parent_section IN (%s, %s))
                """,
                (
                    component.uri,
                    document_id,
                    ascii_number,
                    deva_number,
                    ascii_number,
                    deva_number,
                ),
            )
            linked += int(getattr(cur, "rowcount", 0) or 0)
    return linked


def _lifecycle_bucket_counts(conn: PgConnection) -> Counter[str]:
    counts: Counter[str] = Counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT commencement_dependency, COALESCE(raw_clause_text, ''), COUNT(*)
            FROM lifecycle_effect
            WHERE effect_type='commence'
            GROUP BY commencement_dependency, COALESCE(raw_clause_text, '')
            """
        )
        for dependency, raw_clause_text, count in cur.fetchall():
            counts[_bucket_from_row(dependency, str(raw_clause_text))] += int(count)
    return counts


def backfill_document(
    conn: PgConnection,
    *,
    document_id: str,
    source_id: str,
    content_hash: str,
    records: dict[str, dict[str, Any]],
    dry_run: bool = False,
) -> tuple[str, ParsedLaw | None]:
    record = records.get(source_id)
    if record is None:
        return "missing_record", None

    content = str(record.get("content") or "")
    if _content_hash(content) != content_hash:
        return "hash_mismatch", None

    law = parse_law(record)
    work_id = work_id_for_document(conn, document_id)
    if not work_id:
        return "missing_work_id", None

    stored_uri = work_uri(conn, work_id)
    if stored_uri != law.uri:
        print(
            f"WARNING {source_id}: law.uri != work.uri ({law.uri!r} != {stored_uri!r})",
            file=sys.stderr,
        )
        return "uri_mismatch", None

    if dry_run:
        return "processed", law

    source_pub_id = upsert_source(conn, work_id, law, source_url=None)
    chunk_links = _update_authority_links(
        conn, document_id=document_id, source_pub_id=source_pub_id, law=law
    )
    for component in law.components:
        upsert_component(conn, work_id, component)
        upsert_expression(conn, component, as_of=date.today())
    extract_commencement_proposals(
        law=law, content=content, source_pub_id=source_pub_id, conn=conn
    )
    return f"processed:{chunk_links}", law


def run_backfill(
    conn: PgConnection,
    records: dict[str, dict[str, Any]],
    *,
    dry_run: bool = False,
) -> Summary:
    summary = Summary()
    before = _table_counts(conn) if not dry_run else Counter()
    before_lifecycle = _lifecycle_bucket_counts(conn) if not dry_run else Counter()

    for document_id, source_id, _source_type, content_hash in law_documents(conn):
        try:
            status, law = backfill_document(
                conn,
                document_id=document_id,
                source_id=source_id,
                content_hash=content_hash,
                records=records,
                dry_run=dry_run,
            )
            if status.startswith("processed:"):
                summary.counts["processed"] += 1
                summary.counts["chunk_links_written"] += int(status.split(":", 1)[1])
            else:
                summary.counts[status] += 1
            if law:
                record = records[source_id]
                bucket, rows = _proposal_bucket(law, str(record.get("content") or ""))
                if dry_run:
                    summary.proposal_counts[bucket] += rows
                summary.counts["components_seen"] += len(law.components)
                summary.counts["expressions_seen"] += len(law.components)
                summary.counts["sources_seen"] += 1
            conn.rollback() if dry_run else conn.commit()
        except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the batch
            conn.rollback()
            summary.counts["write_exception"] += 1
            if len(summary.failed_source_ids) < 10:
                summary.failed_source_ids.append(source_id)
            print(f"WARNING {source_id}: backfill failed: {exc}", file=sys.stderr)

    if not dry_run:
        after = _table_counts(conn)
        summary.counts["components_written"] = after["component"] - before["component"]
        summary.counts["sources_written"] = (
            after["source_publication"] - before["source_publication"]
        )
        summary.counts["expressions_written"] = (
            after["expression"] - before["expression"]
        )
        summary.counts["lifecycle_written"] = (
            after["lifecycle_effect"] - before["lifecycle_effect"]
        )
        after_lifecycle = _lifecycle_bucket_counts(conn)
        summary.proposal_counts = after_lifecycle - before_lifecycle
    return summary


def print_summary(summary: Summary, *, dry_run: bool) -> None:
    counts = summary.counts
    mode = "dry-run" if dry_run else "backfill"
    print(f"{mode} complete.")
    print(f"  processed:                {counts.get('processed', 0)}")
    print(f"  skipped missing record:   {counts.get('missing_record', 0)}")
    print(f"  skipped hash mismatch:    {counts.get('hash_mismatch', 0)}")
    print(f"  skipped missing work_id:  {counts.get('missing_work_id', 0)}")
    print(f"  skipped uri mismatch:     {counts.get('uri_mismatch', 0)}")
    print(f"  skipped write exception:  {counts.get('write_exception', 0)}")
    if summary.failed_source_ids:
        print(f"  write exception samples:  {', '.join(summary.failed_source_ids)}")

    if dry_run:
        print(f"  components would write:   {counts.get('components_seen', 0)}")
        print(f"  source pubs would write:  {counts.get('sources_seen', 0)}")
        print(f"  doc source links:         {counts.get('sources_seen', 0)}")
        print(
            f"  chunk authority links:    {counts.get('components_seen', 0)} components scanned"
        )
        print(f"  expressions would write:  {counts.get('expressions_seen', 0)}")
    else:
        print(f"  components written:       {counts.get('components_written', 0)}")
        print(f"  source pubs written:      {counts.get('sources_written', 0)}")
        print(f"  expressions written:      {counts.get('expressions_written', 0)}")
        print(f"  lifecycle rows written:   {counts.get('lifecycle_written', 0)}")
        print(f"  chunk links written:      {counts.get('chunk_links_written', 0)}")

    print("  commencement proposals:")
    for key in sorted(summary.proposal_counts):
        print(f"    {key}: {summary.proposal_counts[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "laws.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_laws(args.input)
    conn = psycopg2.connect(_db_url())
    conn.autocommit = False
    try:
        summary = run_backfill(conn, records, dry_run=args.dry_run)
        print_summary(summary, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
