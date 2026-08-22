"""
Retrieval quality eval: Recall@1/3/5 and MRR.
Uses golden/romanized.json — same queries, matching on document_source_id.

Run: python3 -m app.eval.retrieval_slice
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.authority.writer import connect
from app.retrieval.postgres_retriever import retrieve_postgres

_GOLDEN = Path(__file__).parent / "golden" / "romanized.json"


def run_slice(ks: list[int] | None = None) -> dict[str, Any]:
    if ks is None:
        ks = [1, 3, 5]
    golden: list[dict[str, Any]] = json.loads(_GOLDEN.read_text())
    max_k = max(ks)

    recall: dict[int, int] = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []

    for entry in golden:
        as_of = date.fromisoformat(entry["as_of"])
        expected_sids = {u.split("/")[-1] for u in entry.get("expected_uris", [])}
        with connect() as conn:
            results = retrieve_postgres(conn, entry["query"], as_of, k=max_k)

        result_sids = [r.get("document_source_id", "") for r in results]

        for k in ks:
            if any(sid in expected_sids for sid in result_sids[:k]):
                recall[k] += 1

        rr = 0.0
        for rank, sid in enumerate(result_sids, start=1):
            if sid in expected_sids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(golden)
    return {
        **{f"recall_at_{k}": recall[k] / n for k in ks},
        "mrr": sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0,
        "n": n,
    }


if __name__ == "__main__":
    result = run_slice()
    print(json.dumps(result, indent=2))
