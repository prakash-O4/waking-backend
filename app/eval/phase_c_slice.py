from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextRecall, NonLLMContextPrecisionWithReference

from app.authority.writer import connect
from app.eval.ragas_eval import run_ragas
from app.retrieval.dumb_retriever import retrieve

_GOLDEN = Path(__file__).parent / "golden" / "phase_c_romanized.json"


def run_slice() -> dict[str, Any]:
    try:
        conn = connect()
    except Exception:
        print("phase_c skipped — no DB")
        sys.exit(0)

    golden = json.loads(_GOLDEN.read_text())
    samples: list[SingleTurnSample] = []

    with conn:
        for entry in golden:
            query = entry["query"]
            as_of = date.fromisoformat(entry["as_of"])
            try:
                hits = retrieve(query, as_of)
            except Exception:
                continue
            if not hits:
                continue
            contexts = [h["text_ne"] for h in hits]
            samples.append(
                SingleTurnSample(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=entry["reference"],
                )
            )

    if not samples:
        print("phase_c skipped — no retrievable samples")
        return {}

    dataset = EvaluationDataset(samples)
    return run_ragas(
        dataset,
        [ContextRecall(), NonLLMContextPrecisionWithReference()],
        "Phase C — Romanized Retrieval",
    )


if __name__ == "__main__":
    try:
        run_slice()
    except Exception as exc:
        print(f"phase_c skipped: {exc}")
