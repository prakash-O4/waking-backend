from __future__ import annotations

from datetime import date
from typing import Any, cast

from app.retrieval import query_graph as qg
from app.retrieval.eligibility_gate import eligible_chunk_ids
from tests.stress.helpers import StressConn, chunk, effect


def _conn(parent_effects: list[dict[str, Any]]) -> StressConn:
    conn = StressConn(
        {
            "reg": chunk(
                "/rule/section-1",
                [effect("commence", date(2020, 1, 1))],
                source_type="regulation",
                work_id="reg-work",
            ),
            "enabling": chunk(
                "/act/enabling-55",
                parent_effects,
                work_id="act-work",
                section_number="५५",
                act_name="Enabling Act",
            ),
        }
    )
    conn.relations["reg-work"] = ("act-work", "55", "dafa")
    return conn


def _hit() -> dict[str, Any]:
    return {"component_uri": "reg", "_issue_idx": 2}


def test_enabling_power_returns_none_when_enabling_section_repealed(
    monkeypatch: Any,
) -> None:
    conn = _conn(
        [
            effect("commence", date(2020, 1, 1)),
            effect("repeal", date(2024, 1, 1)),
        ]
    )
    monkeypatch.setattr(qg, "eligible_chunk_ids", eligible_chunk_ids)

    assert qg._fetch_enabling_chunk(cast(Any, conn), _hit(), date(2024, 1, 1)) is None


def test_enabling_power_returns_in_force_enabling_section(monkeypatch: Any) -> None:
    conn = _conn([effect("commence", date(2020, 1, 1))])
    monkeypatch.setattr(qg, "eligible_chunk_ids", eligible_chunk_ids)

    added = qg._fetch_enabling_chunk(cast(Any, conn), _hit(), date(2024, 1, 1))

    assert added is not None
    assert added["component_uri"] == "enabling"
    assert added["section_number"] == "५५"
    assert added["_issue_idx"] == 2
