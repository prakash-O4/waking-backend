from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval import gated_orchestrator as orchestrator
from app.retrieval.eligibility_gate import eligible_chunk_ids
from tests.stress.helpers import StressConn, chunk, effect


def _hit() -> dict[str, Any]:
    return {
        "component_uri": "source",
        "text_ne": "यो दफा ४५६ मा उल्लेख भएको छ",
        "score": 0.8,
        "section_number": "100",
        "document_source_id": "src",
    }


def test_cross_ref_excludes_repealed_target(monkeypatch: Any) -> None:
    conn = StressConn(
        {
            "source": chunk("/law/source", [effect("commence", date(2020, 1, 1))]),
            "target": chunk(
                "/law/target",
                [
                    effect("commence", date(2020, 1, 1)),
                    effect("repeal", date(2024, 1, 1)),
                ],
                section_number="456",
            ),
        }
    )
    monkeypatch.setattr(orchestrator, "eligible_chunk_ids", eligible_chunk_ids)

    assert (
        orchestrator._resolve_cross_refs([_hit()], date(2024, 1, 1), cast(Any, conn))
        == []
    )


def test_cross_ref_adds_eligible_target(monkeypatch: Any) -> None:
    conn = StressConn(
        {
            "source": chunk("/law/source", [effect("commence", date(2020, 1, 1))]),
            "target": chunk(
                "/law/target",
                [effect("commence", date(2020, 1, 1))],
                section_number="456",
            ),
        }
    )
    monkeypatch.setattr(orchestrator, "eligible_chunk_ids", eligible_chunk_ids)

    added = orchestrator._resolve_cross_refs(
        [_hit()], date(2024, 1, 1), cast(Any, conn)
    )

    assert [h["component_uri"] for h in added] == ["target"]
