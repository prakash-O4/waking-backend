#!/usr/bin/env python3
"""Review Langfuse /ask traffic into eval goldens."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.authority.writer import connect  # noqa: E402
from app.retrieval.gated_orchestrator import _structured_claims  # noqa: E402
from app.retrieval.postgres_retriever import get_lf_client, retrieve_postgres  # noqa: E402
from app.retrieval.validation_gate import (  # noqa: E402
    _claim_supported,
    _expression,
    validate_and_render,
)

GOLDEN_DIR = ROOT / "app" / "eval" / "golden"
QUEUE_FILE = GOLDEN_DIR / "_traffic_queue.json"
LABELED_TRAFFIC_FILE = GOLDEN_DIR / "labeled_traffic.json"
CLAIM_SUPPORT_FILE = GOLDEN_DIR / "claim_support.json"


def _read_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def _write_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _is_hash_input(value: str) -> bool:
    return len(value) == 16 and all(c in "0123456789abcdef" for c in value)


def _question(value: Any) -> str:
    if not isinstance(value, str):
        raise SystemExit("trace.input is not a string; refusing to store raw object")
    return value


def _trace_timestamp(trace: Any) -> datetime:
    ts = trace.timestamp
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return cast(datetime, ts)


def fetch(since_days: int, limit: int) -> None:
    client = get_lf_client()
    if client is None:
        raise SystemExit("Langfuse client unavailable; check LANGFUSE_* settings")
    queue = _read_list(QUEUE_FILE)
    seen = {row["trace_id"] for row in queue}
    response = client.api.trace.list(
        name="rag.query",
        from_timestamp=datetime.now(timezone.utc) - timedelta(days=since_days),
        limit=limit,
        order_by="timestamp.desc",
    )
    added = 0
    for trace in response.data:
        trace_id = str(trace.id)
        if trace_id in seen:
            continue
        question = _question(trace.input)
        metadata = trace.metadata or {}
        as_of = metadata.get("as_of")
        as_of_source = "metadata"
        if not as_of:
            as_of = _trace_timestamp(trace).date().isoformat()
            as_of_source = "trace_timestamp_fallback"
        queue.append(
            {
                "trace_id": trace_id,
                "question": question,
                "as_of": str(as_of),
                "as_of_source": as_of_source,
                "content_available": not _is_hash_input(question),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
            }
        )
        seen.add(trace_id)
        added += 1
    _write_list(QUEUE_FILE, queue)
    print(f"queued {added} new candidates")


def _candidate(trace_id: str) -> dict[str, Any]:
    for row in _read_list(QUEUE_FILE):
        if row.get("trace_id") == trace_id:
            if not row.get("content_available"):
                raise SystemExit(
                    "content unavailable for this trace; question text was hashed"
                )
            return row
    raise SystemExit(f"trace not found: {trace_id}")


def list_candidates(status: str) -> None:
    for row in _read_list(QUEUE_FILE):
        if row.get("status") != status:
            continue
        q = str(row.get("question", ""))
        if len(q) > 100:
            q = q[:97] + "..."
        print(f"{row.get('trace_id')}  {row.get('as_of')}  {q}")


def _run_pipeline(
    row: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[bool]
]:
    as_of = date.fromisoformat(str(row["as_of"]))
    with connect() as conn:
        hits = retrieve_postgres(conn, row["question"], as_of)
        claims_doc = (
            _structured_claims(None, [{"query": row["question"], "as_of": as_of}], hits)
            or {}
        )
        claims = list(claims_doc.get("claims", []))
        rendered = validate_and_render(claims, as_of, conn)
        quote_check_passed = []
        for claim in claims:
            expr = _expression(conn, str(claim.get("evidence_id", "")), as_of)
            quote_check_passed.append(
                False
                if expr is None
                else _claim_supported(str(claim.get("quote", "")), expr[0])
            )
    return hits, claims, rendered, quote_check_passed


def show(trace_id: str) -> None:
    hits, claims, rendered, _ = _run_pipeline(_candidate(trace_id))
    print("hits:")
    for hit in hits:
        print(
            f"- {hit.get('component_uri')}  {hit.get('work_title_ne') or hit.get('work_title_en') or ''} "
            f"{hit.get('component_type') or ''} {hit.get('number') or ''} tier={hit.get('tier')}"
        )
    print("claims:")
    for i, claim in enumerate(claims):
        print(
            f"[{i}] claim={claim.get('claim')} evidence_id={claim.get('evidence_id')} quote={claim.get('quote')}"
        )
    print("gate:")
    for i, item in enumerate(rendered):
        print(f"[{i}] abstained={item.get('abstained')} claim={item.get('claim')}")


def _parse_claims(
    raw: list[str] | None, count: int | None = None
) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for item in raw or []:
        try:
            idx_s, verdict = item.split(":", 1)
            idx = int(idx_s)
        except ValueError:
            raise SystemExit(f"invalid --claim: {item}")
        if verdict == "refutes":
            raise SystemExit("use IDX:unsupported, not IDX:refutes")
        if verdict not in {"supports", "unsupported", "skip"}:
            raise SystemExit(f"invalid claim verdict: {verdict}")
        if count is not None and (idx < 0 or idx >= count):
            raise SystemExit(f"claim index out of range: {idx}")
        parsed.append((idx, verdict))
    return parsed


def label(
    trace_id: str, by: str, uris: str | None, raw_claims: list[str] | None
) -> None:
    if not by:
        raise SystemExit("--by is required")
    expected_uris = [u.strip() for u in (uris or "").split(",") if u.strip()]
    parsed_claims = _parse_claims(raw_claims)
    if not expected_uris and not parsed_claims:
        raise SystemExit("at least one of --uris or --claim is required")
    if not expected_uris and all(verdict == "skip" for _, verdict in parsed_claims):
        raise SystemExit(
            "nothing to record — every claim was 'skip' and no --uris given; use --skip if this candidate isn't usable"
        )
    row = _candidate(trace_id)
    _, claims, rendered, quote_check_passed = _run_pipeline(row)
    _parse_claims(raw_claims, len(claims))
    now = datetime.now(timezone.utc).isoformat()
    if expected_uris:
        traffic = _read_list(LABELED_TRAFFIC_FILE)
        traffic.append(
            {
                "query": row["question"],
                "as_of": row["as_of"],
                "expected_uris": expected_uris,
                "source": f"langfuse:{trace_id}",
                "labeled_by": by,
                "labeled_at": now,
            }
        )
        _write_list(LABELED_TRAFFIC_FILE, traffic)
    support_rows = _read_list(CLAIM_SUPPORT_FILE)
    wrote_claim = False
    for idx, verdict in parsed_claims:
        if verdict == "skip":
            continue
        claim = claims[idx]
        gate = rendered[idx] if idx < len(rendered) else {}
        wrote_claim = True
        support_rows.append(
            {
                "query": row["question"],
                "as_of": row["as_of"],
                "quote": claim.get("quote", ""),
                "claim": claim.get("claim", ""),
                "evidence_id": claim.get("evidence_id", ""),
                "supports": verdict == "supports",
                "gate_verdict_abstained": bool(gate.get("abstained")),
                "quote_check_passed": quote_check_passed[idx],
                "source": f"langfuse:{trace_id}",
                "labeled_by": by,
                "labeled_at": now,
            }
        )
    if wrote_claim:
        _write_list(CLAIM_SUPPORT_FILE, support_rows)
    _set_status(trace_id, "labeled")
    print("labeled")


def report() -> None:
    rows = _read_list(CLAIM_SUPPORT_FILE)
    if not rows:
        print("no labeled claims yet")
        return

    counts: dict[tuple[bool, bool], int] = {}
    examples: list[str] = []
    for row in rows:
        key = (bool(row.get("supports")), bool(row.get("quote_check_passed")))
        counts[key] = counts.get(key, 0) + 1
        if key == (False, True) and len(examples) < 5:
            examples.append(str(row.get("source") or row.get("trace_id") or "unknown"))

    for key in [(True, True), (True, False), (False, True), (False, False)]:
        print(f"supports={key[0]}, quote_check_passed={key[1]}: {counts.get(key, 0)}")
    print(f"total: {len(rows)}")
    denominator = counts.get((True, True), 0) + counts.get((False, True), 0)
    if denominator:
        print(f"gap rate: {counts.get((False, True), 0) / denominator:.1%}")
    else:
        print("gap rate: n/a (no quote_check_passed=True rows)")
    if examples:
        print("examples for supports=False, quote_check_passed=True:")
        for example in examples:
            print(f"- {example}")


def _set_status(trace_id: str, status: str, reason: str | None = None) -> None:
    queue = _read_list(QUEUE_FILE)
    for row in queue:
        if row.get("trace_id") == trace_id:
            row["status"] = status
            if reason:
                row["reason"] = reason
            _write_list(QUEUE_FILE, queue)
            return
    raise SystemExit(f"trace not found: {trace_id}")


def skip(trace_id: str, by: str, reason: str) -> None:
    if not by:
        raise SystemExit("--by is required")
    if not reason:
        raise SystemExit("--reason is required")
    _set_status(trace_id, "skipped", reason)
    print("skipped")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--list", action="store_true")
    group.add_argument("--show")
    group.add_argument("--label")
    group.add_argument("--skip")
    group.add_argument("--report", action="store_true")
    parser.add_argument("--since-days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--status", choices=["pending", "labeled", "skipped"], default="pending"
    )
    parser.add_argument("--by")
    parser.add_argument("--uris")
    parser.add_argument("--claim", action="append")
    parser.add_argument("--reason")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.fetch:
        fetch(args.since_days, args.limit)
    elif args.list:
        list_candidates(args.status)
    elif args.show:
        show(args.show)
    elif args.label:
        label(args.label, args.by, args.uris, args.claim)
    elif args.skip:
        skip(args.skip, args.by, args.reason or "")
    elif args.report:
        report()


if __name__ == "__main__":
    main()
