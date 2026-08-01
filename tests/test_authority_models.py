from __future__ import annotations

from datetime import date
from uuid import uuid4

from app.authority.models import Component, ComponentType, Expression, Work, WorkType


def test_authority_models_validate() -> None:
    work_id = uuid4()
    Work(uri="/np/act/2063/demo", work_type=WorkType.ACT, title_ne="ऐन")
    Component(
        work_id=work_id,
        uri="/np/act/2063/demo/dafa/1",
        component_type=ComponentType.DAFA,
    )
    Expression(
        component_uri="/np/act/2063/demo/dafa/1",
        as_of=date(2080, 1, 1),
        text_ne="पाठ",
        text_hash="a" * 64,
    )
