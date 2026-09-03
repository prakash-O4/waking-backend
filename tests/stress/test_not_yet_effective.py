from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval.eligibility_gate import eligible_chunk_ids
from tests.stress.helpers import StressConn, chunk, effect


def test_not_yet_effective_excluded_until_commencement() -> None:
    conn = StressConn(
        {"future": chunk("/law/future", [effect("commence", date(2025, 1, 1))])}
    )

    assert eligible_chunk_ids(cast(Any, conn), date(2024, 12, 31)) == set()
    assert eligible_chunk_ids(cast(Any, conn), date(2025, 1, 1)) == {"future"}


def test_pending_commencement_dependency_stays_excluded() -> None:
    conn = StressConn(
        {
            "pending": chunk(
                "/law/pending",
                [
                    effect(
                        "commence",
                        date(2024, 1, 1),
                        commencement_dependency="gazette_notification",
                    )
                ],
            )
        }
    )

    assert eligible_chunk_ids(cast(Any, conn), date(2025, 1, 1)) == set()
