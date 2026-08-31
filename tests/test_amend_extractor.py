# mypy: ignore-errors
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.authority.writer import propose_lifecycle_amend
from app.ingestion import amend_extractor as ae


CONTENT = """परीक्षण ऐन, २०८०

संशोधन गर्ने ऐन
१. पहिलो संशोधन ऐन, २०८१    २०८१।०१।०२
२. केही नेपाल ऐन संशोधन गर्ने ऐन, २०८२    २०८२।०२।०३

**१. पहिलो दफा:** यो लामो परीक्षण पाठ हो । <amend>पहिलो संशोधनद्वारा संशोधित।</amend> अरु पाठ ।

**२. दोस्रो दफा:** यो पनि लामो परीक्षण पाठ हो । <amend>केही नेपाल ऐन संशोधन गर्ने ऐन, २०८२ द्वारा संशोधित।</amend> अरु पाठ ।
"""


def law():
    return SimpleNamespace(
        uri="/np/act/2080/test",
        components=[
            SimpleNamespace(uri="/np/act/2080/test/full/0"),
            SimpleNamespace(uri="/np/act/2080/test/dafa/1"),
            SimpleNamespace(uri="/np/act/2080/test/dafa/2"),
        ],
    )


def test_parse_amendment_table(monkeypatch):
    monkeypatch.setattr(ae, "lookup", lambda y, m, d: (date(y - 57, m, d), False))
    rows = ae.parse_amendment_table(CONTENT)
    assert [(r.position, r.normalized_name, r.effective_date) for r in rows] == [
        (1, "पहिलो संशोधन ऐन २०८१", date(2024, 1, 2)),
        (2, "केही नेपाल ऐन संशोधन गर्ने ऐन २०८२", date(2025, 2, 3)),
    ]


def test_ordinal_and_named_tags_write_component_proposals(monkeypatch):
    monkeypatch.setattr(ae, "lookup", lambda y, m, d: (date(y - 57, m, d), False))
    calls = []
    monkeypatch.setattr(
        ae, "propose_lifecycle_amend", lambda *a, **kw: calls.append((a, kw))
    )

    result = ae.extract_amend_proposals(
        law=law(), content=CONTENT, source_pub_id="s", conn=None
    )

    assert [p.method for p in result.proposals] == ["ordinal", "named-act"]
    assert [c[0][1] for c in calls] == [
        "/np/act/2080/test/dafa/1",
        "/np/act/2080/test/dafa/2",
    ]
    assert calls[0][1]["effective_date"] == date(2024, 1, 2)
    assert calls[1][1]["effective_date"] == date(2025, 2, 3)
    assert result.skipped == {}


def test_skips_without_guessing(monkeypatch):
    monkeypatch.setattr(ae, "lookup", lambda y, m, d: (date(y - 57, m, d), False))
    content = CONTENT + "\n**३. तेस्रो दफा:** लामो पाठ । <amend>२०८१।०१।०२</amend>"
    result = ae.extract_amend_proposals(
        law=law(), content=content, source_pub_id="s", conn=MagicMock()
    )
    assert result.skipped["unresolved_gazette-date-only"] == 1


def test_no_table_skips_all():
    content = "**१. दफा:** लामो पाठ । <amend>पहिलो संशोधनद्वारा संशोधित।</amend>"
    result = ae.extract_amend_proposals(
        law=law(), content=content, source_pub_id="s", conn=MagicMock()
    )
    assert result.proposals == []
    assert result.skipped == {"no_table": 1}


def test_writer_dedups_on_component_and_raw_clause():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (1,)
    propose_lifecycle_amend(
        conn,
        "/c/1",
        "00000000-0000-0000-0000-000000000001",
        effective_date=None,
        amendment_dependency="amendment_date_unresolved:२०८१।१।१",
        raw_clause_text="पहिलो संशोधनद्वारा संशोधित।",
    )
    assert cur.execute.call_count == 1


def test_writer_inserts_empty_valid_time_when_date_missing():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    propose_lifecycle_amend(
        conn,
        "/c/1",
        "00000000-0000-0000-0000-000000000001",
        effective_date=None,
        amendment_dependency="amendment_date_unresolved:२०८१।१।१",
        raw_clause_text="पहिलो संशोधनद्वारा संशोधित।",
    )
    insert_params = cur.execute.call_args_list[-1].args[1]
    assert insert_params[1] == "empty"
    assert insert_params[2] is None
    assert insert_params[3] == "amendment_date_unresolved:२०८१।१।१"
