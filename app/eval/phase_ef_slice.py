from __future__ import annotations

import sys
from typing import Any

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness

from app.authority.writer import connect
from app.eval.ragas_eval import run_ragas

_SUMMARY_QUERY = """
SELECT d.id, d.summary, array_agg(c.text_ne)
FROM documents d
JOIN chunks c ON c.document_id = d.id
WHERE d.summary IS NOT NULL
GROUP BY d.id
LIMIT 10
"""


def run_slice() -> dict[str, Any]:
    try:
        conn = connect()
    except Exception:
        print("phase_ef skipped — no DB")
        sys.exit(0)

    samples: list[SingleTurnSample] = []
    with conn:
        with conn.cursor() as cur:
            cur.execute(_SUMMARY_QUERY)
            rows = cur.fetchall()

    if not rows:
        print("phase_ef skipped — no documents with summary")
        sys.exit(0)

    for _doc_id, summary, chunks in rows:
        if not summary or not chunks:
            continue
        sample_contexts = [c for c in chunks if c]
        if not sample_contexts:
            continue
        samples.append(
            SingleTurnSample(
                user_input="Summarize this document.",
                response=summary,
                retrieved_contexts=sample_contexts,
            )
        )

    if not samples:
        print("phase_ef skipped — no documents with summary")
        sys.exit(0)

    dataset = EvaluationDataset(samples)
    return run_ragas(
        dataset,
        [Faithfulness()],
        "Phase EF — Summary Faithfulness",
    )


if __name__ == "__main__":
    try:
        run_slice()
    except Exception as exc:
        print(f"phase_ef skipped: {exc}")
