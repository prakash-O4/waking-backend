from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval.eligibility_gate import eligible_chunk_ids
from app.retrieval.validation_gate import validate_and_render
from tests.stress.helpers import QUOTE, StressConn, chunk, effect


def test_repealed_current_pre_retrieval_and_validation_gate() -> None:
    conn = StressConn(
        {
            "chunk-1": chunk(
                "/law/section-1",
                [
                    effect("commence", date(2020, 1, 1)),
                    effect("repeal", date(2024, 1, 1)),
                ],
            )
        }
    )
    claim = {"claim": "ok", "evidence_id": "chunk-1", "quote": QUOTE}

    assert eligible_chunk_ids(cast(Any, conn), date(2023, 12, 31)) == {"chunk-1"}
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()

    before = validate_and_render([claim], date(2023, 12, 31), cast(Any, conn))[0]
    after = validate_and_render([claim], date(2024, 1, 1), cast(Any, conn))[0]

    assert before["citation"] is not None
    assert before["abstained"] is False
    assert after["citation"] is None
    assert after["abstained"] is True
