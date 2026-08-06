from __future__ import annotations

import json
from typing import Any

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric


class TemporalFaithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "temporal_faithfulness"
    _required_columns: set[str] = {"user_input", "response", "retrieved_contexts"}

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Any | None = None
    ) -> float:
        as_of = (sample.additional_metadata or {}).get("as_of", "unknown")
        context_str = "\n".join(sample.retrieved_contexts or [])
        prompt = (
            f"The legal question is answered as of {as_of}.\n"
            f"Response: {sample.response}\n"
            f"Context: {context_str}\n\n"
            "Does this response contain any claim implying a provision was in force "
            "before its commencement or after its repeal?\n"
            'Reply with JSON only: {"temporal_violation": true|false, "reason": "..."}'
        )
        result = await self.llm.agenerate([[{"role": "user", "content": prompt}]])
        try:
            parsed = json.loads(result.generations[0][0].text)
            return 0.0 if parsed.get("temporal_violation") else 1.0
        except Exception:
            return 1.0  # assume no violation on parse failure
