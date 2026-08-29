"""
Unit tests for enabling-power extraction from regulation preambles.

No LLM is mocked or called — the extractor is fully deterministic.
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ingestion import enabling_extractor as ee


def test_enabling_regex_standard() -> None:
    preamble = (
        "यस नियमावलीलाई सुशासन (व्यवस्थापन तथा सञ्चालन) ऐन, २०६४ को "
        "दफा ५५ ले दिएको अधिकार प्रयोग गरी बनाइएको हो ।"
    )
    row = ee._parse_enabling_clause(preamble)
    assert row is not None
    (
        provision_type_raw,
        section_raw,
        subsection_raw,
        act_ref,
        provision_kind,
        raw_match,
    ) = row
    assert provision_type_raw == "दफा"
    assert provision_kind == "dafa"
    assert section_raw == "५५"
    assert subsection_raw is None
    assert "सुशासन (व्यवस्थापन तथा सञ्चालन) ऐन, २०६४" in act_ref
    assert "ले दिएको अधिकार" in raw_match


def test_enabling_regex_upadafa_variant() -> None:
    preamble = (
        "यस नियमावलीलाई कर्मचारी समायोजन ऐन, २०७५ को दफा १० को उपदफा (२) ले "
        "दिएको अधिकार प्रयोग गरी बनाइएको हो ।"
    )
    row = ee._parse_enabling_clause(preamble)
    assert row is not None
    (
        provision_type_raw,
        section_raw,
        subsection_raw,
        act_ref,
        provision_kind,
        raw_match,
    ) = row
    assert provision_type_raw == "दफा"
    assert provision_kind == "dafa"
    assert section_raw == "१०"
    assert subsection_raw == "२"
    assert "कर्मचारी समायोजन ऐन, २०७५" in act_ref
    assert "उपदफा (२)" in raw_match


def test_enabling_regex_amend_markup_no_false_positive() -> None:
    # A stray closing amend tag can create a false "act reference" for the regex.
    text = "</amend>को दफा ५५ ले दिएको अधिकार प्रयोग गरी बनाइएको हो ।"
    stripped = ee._strip_amend_markup(text)
    assert "<amend>" not in stripped
    assert "</amend>" not in stripped
    # After stripping, there is no 5+ char act-reference before "को".
    assert ee._parse_enabling_clause(stripped) is None


def test_title_normalization_comma() -> None:
    assert ee._normalize_title("लोक सेवा आयोग ऐन, २०७९") == "लोक सेवा आयोग ऐन २०७९"


def test_section_normalization_devanagari() -> None:
    assert ee._normalize_section_num("४४") == "44"
    assert ee._normalize_section_num("१०") == "10"
    assert ee._normalize_section_num("(२)") == "(2)"


def test_extract_inserts_resolved_row(monkeypatch: pytest.MonkeyPatch) -> None:
    content = (
        "परिच्छेद-१\n\nयस नियमावलीलाई सुशासन (व्यवस्थापन तथा सञ्चालन) ऐन, २०६४ "
        "ले केही गरेको छ । सुशासन ऐन, २०६४ को दफा ५५ ले दिएको अधिकार प्रयोग "
        "गरी बनाइएको हो ।\n\n**१. संक्षिप्त नाम:**"
    )
    parent_id = str(uuid.uuid4())
    calls: list[dict[str, Any]] = []

    def fake_resolve(_conn: Any, title: str) -> str | None:
        assert title == "सुशासन ऐन २०६४"
        return parent_id

    def fake_insert(
        _conn: Any,
        subordinate_work_id: str,
        enabling_work_id: str | None,
        provision_type: str | None,
        section_num: str | None,
        subsection_num: str | None,
        raw_clause_text: str,
        status: str,
    ) -> None:
        calls.append(
            {
                "subordinate_work_id": subordinate_work_id,
                "enabling_work_id": enabling_work_id,
                "provision_type": provision_type,
                "section_num": section_num,
                "subsection_num": subsection_num,
                "raw_clause_text": raw_clause_text,
                "status": status,
            }
        )

    monkeypatch.setattr(ee, "_resolve_work", fake_resolve)
    monkeypatch.setattr(ee, "_insert_relation", fake_insert)

    work_id = str(uuid.uuid4())
    ee.extract_enabling_clause(content, work_id, MagicMock())

    assert len(calls) == 1
    call = calls[0]
    assert call["subordinate_work_id"] == work_id
    assert call["enabling_work_id"] == parent_id
    assert call["provision_type"] == "dafa"
    assert call["section_num"] == "55"
    assert call["subsection_num"] is None
    assert call["status"] == "auto_extracted"
    assert "सुशासन ऐन, २०६४" in call["raw_clause_text"]
    assert "दफा" in call["raw_clause_text"]
    assert "ले दिएको अधिकार" in call["raw_clause_text"]


def test_extract_inserts_parent_not_in_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    content = (
        "यस नियमावलीलाई अस्तित्वमा नभएको ऐन, २०७७ को दफा ३ ले दिएको अधिकार "
        "प्रयोग गरी बनाइएको हो ।"
    )
    monkeypatch.setattr(ee, "_resolve_work", lambda _conn, _title: None)

    calls: list[dict[str, Any]] = []

    def fake_insert(
        _conn: Any,
        subordinate_work_id: str,
        enabling_work_id: str | None,
        provision_type: str | None,
        section_num: str | None,
        subsection_num: str | None,
        raw_clause_text: str,
        status: str,
    ) -> None:
        calls.append(
            {
                "enabling_work_id": enabling_work_id,
                "provision_type": provision_type,
                "section_num": section_num,
                "status": status,
            }
        )

    monkeypatch.setattr(ee, "_insert_relation", fake_insert)
    ee.extract_enabling_clause(content, str(uuid.uuid4()), MagicMock())

    assert len(calls) == 1
    assert calls[0]["enabling_work_id"] is None
    assert calls[0]["status"] == "parent_not_in_corpus"


def test_extract_inserts_no_clause_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    content = "यस नियमावलीमा कुनै enabling clause छैन ।"
    calls: list[dict[str, Any]] = []

    def fake_insert(
        _conn: Any,
        subordinate_work_id: str,
        enabling_work_id: str | None,
        provision_type: str | None,
        section_num: str | None,
        subsection_num: str | None,
        raw_clause_text: str,
        status: str,
    ) -> None:
        calls.append({"status": status, "raw_clause_text": raw_clause_text})

    monkeypatch.setattr(ee, "_insert_relation", fake_insert)
    ee.extract_enabling_clause(content, str(uuid.uuid4()), MagicMock())

    assert len(calls) == 1
    assert calls[0]["status"] == "no_enabling_clause"
    assert calls[0]["raw_clause_text"] == ""


def test_extract_idempotent_sql_has_conflict_clause() -> None:
    """The insert must be idempotent at the DB level (ON CONFLICT DO NOTHING)."""
    content = (
        "यस नियमावलीलाई सुशासन ऐन, २०६४ को दफा ५५ ले दिएको अधिकार प्रयोग गरी "
        "बनाइएको हो ।"
    )
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None

    ee.extract_enabling_clause(content, str(uuid.uuid4()), conn)
    ee.extract_enabling_clause(content, str(uuid.uuid4()), conn)

    execute_calls = [call for call in cursor.method_calls if call[0] == "execute"]
    sqls = [str(call[1][0]) for call in execute_calls]
    assert any("ON CONFLICT DO NOTHING" in sql for sql in sqls)


@pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="SUPABASE_DB_URL not set")
def test_extract_idempotent_live_db() -> None:
    """Calling the extractor twice for the same work_id leaves exactly one row."""
    import psycopg2

    content = (
        "यस नियमावलीलाई सुशासन ऐन, २०६४ को दफा ५५ ले दिएको अधिकार प्रयोग गरी "
        "बनाइएको हो ।"
    )
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO work (uri, work_type, title_ne) VALUES (%s, %s, %s) "
                "ON CONFLICT (uri) DO UPDATE SET title_ne = EXCLUDED.title_ne "
                "RETURNING id",
                ("/np/test/enabling-idempotent", "rule", "Enabling Idempotent Test"),
            )
            row = cur.fetchone()
            assert row is not None
            subordinate_id = str(row[0])
            cur.execute(
                "INSERT INTO work (uri, work_type, title_ne) VALUES (%s, %s, %s) "
                "ON CONFLICT (uri) DO UPDATE SET title_ne = EXCLUDED.title_ne "
                "RETURNING id",
                ("/np/test/enabling-parent", "act", "सुशासन ऐन २०६४"),
            )
            conn.commit()

        ee.extract_enabling_clause(content, subordinate_id, conn)
        conn.commit()
        ee.extract_enabling_clause(content, subordinate_id, conn)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM work_relations WHERE subordinate_work_id = %s",
                (subordinate_id,),
            )
            count_row = cur.fetchone()
            assert count_row is not None
            assert count_row[0] == 1
    finally:
        conn.rollback()
        conn.close()
