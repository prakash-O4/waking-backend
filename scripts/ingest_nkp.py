#!/usr/bin/env python3
"""
Ingest NKP Supreme Court cases through the PE-A ingestion pipeline.

Usage:
    python scripts/ingest_nkp.py --input output/nkp_cases.jsonl [--limit N] [--dry-run]

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

from app.ingestion.nkp_chunker import NKPChunker  # noqa: E402
from app.ingestion.pii_redactor import PIIRedactor, RedactionVerificationError  # noqa: E402


def _iter_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]
    return [json.loads(line) for line in lines if line.strip()]


def _dry_run(records: list[dict[str, Any]]) -> None:
    """Redact (deterministic pass only) + chunk locally; no DB, no LLM."""
    redactor = PIIRedactor(enable_llm=False)
    chunker = NKPChunker()
    valid = rejected = quarantined = 0
    for record in records:
        case_id = str(record.get("case_id") or "?")
        full_text = str(record.get("full_text") or "")
        if not full_text.strip():
            rejected += 1
            print(f"{case_id}: REJECTED (empty full_text)")
            continue
        try:
            redacted, _warnings = redactor.redact(
                full_text,
                str(record.get("appellant") or ""),
                str(record.get("respondent") or ""),
                document_id=case_id,
            )
        except RedactionVerificationError as exc:
            quarantined += 1
            print(f"{case_id}: QUARANTINED ({exc})")
            continue
        chunks = chunker.chunk_text(redacted)
        valid += 1
        print(f"{case_id}: {len(chunks)} chunks")
    print(
        f"dry-run: {valid} valid / {rejected} rejected / {quarantined} "
        f"quarantined of {len(records)} records (no DB writes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output" / "nkp_cases.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = _iter_records(args.input, args.limit)
    if args.dry_run:
        _dry_run(records)
        return

    from app.authority.writer import connect
    from app.ingestion.pipeline import IngestionPipeline

    counts = {"processed": 0, "skipped": 0, "rejected": 0, "failed": 0}
    with connect() as conn:
        pipeline = IngestionPipeline(conn)
        for record in records:
            case_id = str(record.get("case_id") or "?")
            try:
                document_id = pipeline.ingest_nkp_case(record)
            except Exception as exc:  # noqa: BLE001 - keep the batch moving
                conn.rollback()
                counts["failed"] += 1
                print(f"error: {case_id} failed: {exc}")
                continue
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


if __name__ == "__main__":
    main()
