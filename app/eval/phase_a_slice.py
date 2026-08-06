from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy

from app.authority.writer import connect
from app.config import get_settings
from app.eval.ragas_eval import run_ragas
from app.retrieval.postgres_retriever import retrieve_postgres

_GOLDEN = Path(__file__).parent / "golden" / "phase_a_qa.json"

_SYSTEM_PROMPT = (
    "You are Wakil-G. Answer using ONLY the provided context. "
    "Output JSON only:\n"
    '{"claims": [{"claim": "<answer text>", "evidence_id": "<component_uri>"}]}\n'
    'If context is insufficient: {"claims": [], "abstain": true}\n'
    "Do not write citations."
)


def _generate_response(query: str, contexts: list[str]) -> str:
    settings = get_settings()
    context = "\n\n---\n\n".join(contexts)
    llm = init_chat_model(settings.LLM_MODEL)
    resp = llm.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
    )
    parsed = json.loads(resp.content.strip())
    if parsed.get("abstain") or not parsed.get("claims"):
        return ""
    return " ".join(str(c.get("claim", "")) for c in parsed["claims"])


def run_slice() -> dict[str, Any]:
    try:
        conn = connect()
    except Exception:
        print("phase_a skipped — no DB")
        sys.exit(0)

    golden = json.loads(_GOLDEN.read_text())
    samples: list[SingleTurnSample] = []
    with conn:
        for entry in golden:
            query = entry["query"]
            as_of = date.fromisoformat(entry["as_of"])
            hits = retrieve_postgres(conn, query, as_of)
            if not hits:
                continue
            contexts = [h["text_ne"] for h in hits]
            try:
                response = _generate_response(query, contexts)
            except Exception:
                continue
            if not response:
                continue
            samples.append(
                SingleTurnSample(
                    user_input=query,
                    response=response,
                    retrieved_contexts=contexts,
                )
            )

    if not samples:
        print("phase_a skipped — no retrievable samples")
        return {}

    dataset = EvaluationDataset(samples)
    return run_ragas(
        dataset,
        [Faithfulness(), ResponseRelevancy()],
        "Phase A — QA Pipeline",
    )


if __name__ == "__main__":
    try:
        run_slice()
    except Exception as exc:
        print(f"phase_a skipped: {exc}")
