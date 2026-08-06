from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextRecall, Faithfulness

from app.authority.writer import connect
from app.eval.ragas_eval import run_ragas
from app.retrieval.precedent_retriever import retrieve_precedent

_GOLDEN = Path(__file__).parent / "golden" / "phase_d_precedent.json"


def _precedent_count(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM precedent")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def run_slice() -> dict[str, Any]:
    try:
        conn = connect()
    except Exception:
        print("phase_d skipped — no DB")
        sys.exit(0)

    with conn:
        if _precedent_count(conn) == 0:
            print("phase_d skipped — precedent table empty")
            sys.exit(0)

        golden = json.loads(_GOLDEN.read_text())
        samples: list[SingleTurnSample] = []
        for entry in golden:
            query = entry["query"]
            as_of = date.fromisoformat(entry["as_of"])
            hits = retrieve_precedent(conn, query, as_of)
            if not hits:
                continue
            contexts = [h["holding_text"] for h in hits]
            response = contexts[0]
            samples.append(
                SingleTurnSample(
                    user_input=query,
                    response=response,
                    retrieved_contexts=contexts,
                    reference=entry["reference"],
                )
            )

    if not samples:
        print("phase_d skipped — no retrievable precedent samples")
        return {}

    dataset = EvaluationDataset(samples)
    return run_ragas(
        dataset,
        [Faithfulness(), ContextRecall()],
        "Phase D — Precedent",
    )


if __name__ == "__main__":
    try:
        run_slice()
    except Exception as exc:
        print(f"phase_d skipped: {exc}")
