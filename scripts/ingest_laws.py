#!/usr/bin/env python3
"""
Ingest laws.jsonl through the PE-A ingestion pipeline (PostgreSQL + pgvector).

Usage:
    python scripts/ingest_laws.py --input laws.jsonl [--limit N] [--dry-run]

Reports: processed / skipped / rejected / failed counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.laws_chunker import LawsChunker  # noqa: E402


def _iter_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]
    return [json.loads(line) for line in lines if line.strip()]


def _dry_run(records: list[dict[str, Any]]) -> None:
    """Validate + chunk locally; no DB connection, no writes."""
    chunker = LawsChunker()
    valid = rejected = 0
    for record in records:
        content = str(record.get("content") or "")
        chunks = chunker.chunk_text(content) if content.strip() else []
        if chunks:
            valid += 1
            print(f"{record.get('_id')}: {len(chunks)} chunks")
        else:
            rejected += 1
            print(f"{record.get('_id')}: REJECTED (empty or no दफा anchor)")
    print(
        f"dry-run: {valid} valid / {rejected} rejected "
        f"of {len(records)} records (no DB writes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "laws.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = _iter_records(args.input, args.limit)
    if args.dry_run:
        _dry_run(records)
        return

    from langchain_community.callbacks import get_openai_callback

    from app.authority.writer import connect
    from app.ingestion.pipeline import IngestionPipeline

    counts = {"processed": 0, "skipped": 0, "rejected": 0, "failed": 0}
    total = len(records)
    print(f"starting ingestion of {total} records…")
    with get_openai_callback() as cb:
        with connect() as conn:
            pipeline = IngestionPipeline(conn)
            for i, record in enumerate(records, 1):
                source_id = str(record.get("_id") or record.get("name") or "?")
                print(f"[{i}/{total}] {source_id} …", flush=True)
                try:
                    document_id = pipeline.ingest_law(record)
                except Exception as exc:  # noqa: BLE001 - keep the batch moving
                    conn.rollback()
                    counts["failed"] += 1
                    print(f"  ✗ failed: {exc}", flush=True)
                    continue
                outcome = pipeline.last_outcome if document_id is None else "ingested"
                print(f"  ✓ {outcome}", flush=True)
                if document_id is not None:
                    counts["processed"] += 1
                elif pipeline.last_outcome == "skipped":
                    counts["skipped"] += 1
                else:
                    counts["rejected"] += 1

    print(
        "done: "
        f"processed={counts['processed']} skipped={counts['skipped']} "
        f"rejected={counts['rejected']} failed={counts['failed']}"
    )
    print(
        f"\n=== ingestion cost ===\n"
        f"  llm tokens   : {cb.prompt_tokens:,} in + {cb.completion_tokens:,} out"
        f" = {cb.total_tokens:,} total\n"
        f"  embed tokens : 0 (bge-m3 local)\n"
        f"  est. cost    : ${cb.total_cost:.4f} USD"
    )


if __name__ == "__main__":
    main()
