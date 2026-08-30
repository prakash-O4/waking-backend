# mypy: ignore-errors
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.authority.writer import propose_lifecycle_commence
from app.ingestion import commencement_extractor as ce


def law(enactment_ad: date | None = date(2020, 1, 1), n: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        uri="/np/act/1/test",
        enactment_ad=enactment_ad,
        components=[SimpleNamespace(uri=f"/np/act/1/test/dafa/{i}") for i in range(n)],
    )


def capture(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ce, "propose_lifecycle_commence", fake)
    return calls


def test_immediate_proposal_per_component(monkeypatch):
    calls = capture(monkeypatch)
    ce.extract_commencement_proposals(
        law=law(), content="यो ऐन तुरुन्त प्रारम्भ हुनेछ", source_pub_id="s", conn=None
    )
    assert len(calls) == 2
    assert all(c[1]["effective_date"] == date(2020, 1, 1) for c in calls)
    assert all(c[1]["commencement_dependency"] is None for c in calls)


def test_immediate_without_enactment_date_is_flagged():
    row = ce.classify_commencement(law(None), "यो ऐन तुरुन्त प्रारम्भ हुनेछ")
    assert row.effective_date is None
    assert row.commencement_dependency == "enactment_date_unknown"


def test_relative_known_ordinal():
    row = ce.classify_commencement(
        law(), "यो ऐन प्रमाणीकरण भएको एकतीसौँ दिनदेखि प्रारम्भ हुनेछ"
    )
    assert row.effective_date == date(2020, 2, 1)
    assert row.commencement_dependency is None


def test_relative_unknown_ordinal_is_pending_unknown():
    row = ce.classify_commencement(
        law(), "यो ऐन प्रमाणीकरण भएको झिलिमिलीऔं दिनदेखि प्रारम्भ हुनेछ"
    )
    assert row.effective_date is None
    assert row.commencement_dependency == "unparsed_relative_delay"
    assert "झिलिमिलीऔं" in row.raw_clause_text


def test_gazette_dependent_and_writer_empty_range():
    row = ce.classify_commencement(
        law(),
        "यो ऐन नेपाल सरकारले नेपाल राजपत्रमा सूचना प्रकाशन गरी तोकेको मितिदेखि प्रारम्भ हुनेछ",
    )
    assert row.effective_date is None
    assert row.commencement_dependency == "gazette_notification_pending"

    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    propose_lifecycle_commence(
        conn,
        "/c/1",
        "00000000-0000-0000-0000-000000000001",
        effective_date=None,
        commencement_dependency=row.commencement_dependency,
        raw_clause_text=row.raw_clause_text,
    )
    insert_params = cur.execute.call_args_list[-1].args[1]
    assert insert_params[1] == "empty"


def test_false_positive_prescribed_definition_not_gazette():
    row = ce.classify_commencement(
        law(),
        "तोकिएको भन्नाले नेपाल सरकारले नेपाल राजपत्रमा सूचना प्रकाशन गरी तोकिएको सम्झनुपर्छ ।",
    )
    assert row.commencement_dependency == "no_commencement_clause"


def test_publication_pattern_with_date():
    row = ce.classify_commencement(
        law(), "यो नियमावली नेपाल राजपत्रमा प्रकाशन भएको मितिदेखि प्रारम्भ हुनेछ"
    )
    assert row.effective_date == date(2020, 1, 1)
    assert row.commencement_dependency is None


def test_publication_pattern_without_date():
    row = ce.classify_commencement(
        law(None), "यो नियमावली नेपाल राजपत्रमा प्रकाशन भएको मितिदेखि प्रारम्भ हुनेछ"
    )
    assert row.effective_date is None
    assert row.commencement_dependency == "publication_date_unknown"


def test_no_pattern_sentinel():
    row = ce.classify_commencement(law(), "कुनै प्रारम्भ दफा छैन")
    assert row.effective_date is None
    assert row.commencement_dependency == "no_commencement_clause"
    assert row.raw_clause_text == ""


def test_no_pattern_sentinel_writes_once_for_work(monkeypatch):
    calls = capture(monkeypatch)
    ce.extract_commencement_proposals(
        law=law(n=3), content="कुनै प्रारम्भ दफा छैन", source_pub_id="s", conn=None
    )
    assert len(calls) == 1
    assert calls[0][0][1] == "/np/act/1/test"


def test_writer_dedups_pending():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (1,)
    propose_lifecycle_commence(
        conn,
        "/c/1",
        "00000000-0000-0000-0000-000000000001",
        effective_date=date(2020, 1, 1),
        commencement_dependency=None,
        raw_clause_text="x",
    )
    assert cur.execute.call_count == 1


def test_same_proposal_for_every_component(monkeypatch):
    calls = capture(monkeypatch)
    ce.extract_commencement_proposals(
        law=law(n=3),
        content="यो ऐन नेपाल सरकारले नेपाल राजपत्रमा सूचना प्रकाशित गरी तोकिदिएको मितिमा प्रारम्भ हुनेछ",
        source_pub_id="s",
        conn=None,
    )
    assert len(calls) == 3
    assert {c[1]["commencement_dependency"] for c in calls} == {
        "gazette_notification_pending"
    }
    assert len({c[1]["raw_clause_text"] for c in calls}) == 1
