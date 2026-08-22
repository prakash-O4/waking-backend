from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.authority.writer import connect
from app.retrieval.postgres_retriever import retrieve_postgres

_GOLDEN = Path(__file__).parent / "golden" / "romanized.json"


def _source_id(hit: dict[str, Any]) -> str:
    sid = hit.get("document_source_id", "")
    if sid:
        return str(sid)
    uri = str(hit.get("component_uri", ""))
    parts = uri.split("/")
    return parts[4] if len(parts) > 4 and parts[:3] == ["", "np", "act"] else ""


def run_slice(k: int = 5) -> dict[str, Any]:
    """Run romanized Nepali golden queries through dumb retrieval."""
    golden = json.loads(_GOLDEN.read_text())
    hits = 0
    for entry in golden:
        as_of = date.fromisoformat(entry["as_of"])
        expected_sids = {u.split("/")[-1] for u in entry.get("expected_uris", [])}
        with connect() as conn:
            results = retrieve_postgres(conn, entry["query"], as_of, k=k)
        result_sids = [_source_id(r) for r in results]
        if any(sid in expected_sids for sid in result_sids):
            hits += 1
    n = len(golden)
    return {"recall_at_k": hits / n if n else 0.0, "k": k, "n_queries": n, "hits": hits}


if __name__ == "__main__":
    try:
        result = run_slice()
        print(
            f"romanized-recall@{result['k']}: "
            f"{result['hits']}/{result['n_queries']} = {result['recall_at_k']:.2f}"
        )
    except Exception as exc:
        print(f"romanized slice skipped: {exc}")
