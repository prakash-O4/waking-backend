from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval import gated_orchestrator as orchestrator
from app.retrieval import query_graph as qg
from app.retrieval.eligibility_gate import eligible_chunk_ids
from tests.stress.helpers import StressConn, chunk, effect


def _hit(chunk_id: str) -> dict[str, Any]:
    return {
        "component_uri": chunk_id,
        "text_ne": "स्पष्टीकरण text",
        "text_hash": "hash",
        "score": 0.9,
        "work_title_ne": "Child Act",
        "chunk_type": "proviso",
        "section_number": "१८",
        "document_source_id": "src",
        "_issue_idx": 0,
    }


def test_proviso_adds_only_in_force_parent_and_keeps_original_hits(
    monkeypatch: Any,
) -> None:
    conn = StressConn(
        {
            "child-live": chunk(
                "/law/child-live",
                [effect("commence", date(2020, 1, 1))],
                parent_id="parent-live",
            ),
            "child-dead": chunk(
                "/law/child-dead",
                [effect("commence", date(2020, 1, 1))],
                parent_id="parent-dead",
            ),
            "parent-live": chunk(
                "/law/parent-live", [effect("commence", date(2020, 1, 1))]
            ),
            "parent-dead": chunk(
                "/law/parent-dead",
                [
                    effect("commence", date(2020, 1, 1)),
                    effect("repeal", date(2024, 1, 1)),
                ],
            ),
        }
    )
    monkeypatch.setattr(orchestrator, "eligible_chunk_ids", eligible_chunk_ids)

    hits = [_hit("child-live"), _hit("child-dead")]
    result = qg.co_retrieve_parent_resolver_node(
        {"session_as_of": date(2024, 1, 1), "all_hits": hits},
        {"configurable": {"conn": cast(Any, conn)}},
    )

    assert [h["component_uri"] for h in result["all_hits"]] == [
        "child-live",
        "child-dead",
        "parent-live",
    ]
    assert hits == result["all_hits"][:2]
