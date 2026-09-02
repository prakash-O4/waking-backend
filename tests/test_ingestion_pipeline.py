"""
Unit tests for the PE-A ingestion pipeline (task.md §Tests).

DB and haiku calls are mocked; no real APIs or databases are hit. The live-DB
dual-approval test is skipped unless SUPABASE_DB_URL points at a Postgres.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ingestion import metadata_enricher
from app.ingestion.laws_chunker import LawChunk, LawsChunker
from app.ingestion.nkp_chunker import NKPChunker
from app.ingestion.pii_redactor import PIIRedactor, RedactionVerificationError
from app.ingestion.pipeline import IngestionPipeline, _content_hash

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_005 = ROOT / "migrations" / "005_ingestion_pipeline.sql"
LAWS_JSONL = ROOT / "laws.jsonl"

FILLER = "यस ऐनको प्रयोजनको लागि परीक्षण पाठ हो जसले दफा लामो बनाउँछ । "

LAW_FIXTURE = f"""परीक्षण ऐन, २०८०

प्रमाणीकरण र प्रकाशन मिति
२०८०।०१।०१

**१. संक्षिप्त नाम र प्रारम्भ:** (१) यस ऐनको नाम "परीक्षण ऐन, २०८०" रहेको छ । <amend>परीक्षण संशोधन ऐन द्वारा संशोधित।</amend>

**२. विस्तृत प्रावधान:** विषय वा प्रसङ्गले अर्को अर्थ नलागेमा यस ऐनमा,-

(१) {FILLER * 30}

(२) {FILLER * 30}

स्पष्टीकरण : यस दफाको प्रयोजनको लागि परीक्षण स्पष्टीकरण पाठ हो । {FILLER * 5}

**३. अन्त्य:** यो ऐन तुरुन्त प्रारम्भ हुनेछ ।
"""

NKP_FIXTURE = """सर्वोच्च अदालत, संयुक्त इजलास

माननीय न्यायाधीश श्री परीक्षण न्यायाधीश

यस मुद्दामा स्थापित सिद्धान्त परीक्षणको लागि हो । छुट दर्ताको प्रक्रिया चालु राख्ने विधिकर्ताको मनसाय होइन ।

न्या.परीक्षण न्यायाधीश : प्रस्तुत मुद्दाको संक्षिप्त तथ्य एवं विवेचना यसप्रकार छ । वादीहरूको जिकिर अनुसार विवादित जग्गाको दर्ता बदर हुनुपर्ने देखिन आयो ।

१. फिराद दाबी नपुग्ने गरी फैसला गर्नुपर्ने ठहर्छ ।

२. प्रस्तुत फैसलाको विद्युतीयप्रति सफ्टवेयरमा अपलोड गर्नु ।

इति संवत् २०८१ साल वैशाख ३ गते रोज २ शुभम्
"""


def _real_law_record(source_id: str) -> dict[str, Any]:
    for line in LAWS_JSONL.read_text(encoding="utf-8").splitlines():
        record: dict[str, Any] = json.loads(line)
        if record.get("_id") == source_id:
            return record
    raise AssertionError(f"missing laws.jsonl record {source_id}")


def test_laws_chunker_structure() -> None:
    chunks = LawsChunker().chunk_text(LAW_FIXTURE)

    levels = [c.level for c in chunks]
    assert levels == [
        "act",
        "section",
        "subsection",
        "subsection",
        "subsection",
        "proviso",
        "section",
    ]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # Preamble → act chunk; दफा १ stays whole (< 2,400 chars).
    assert chunks[0].section_number is None
    assert "प्रमाणीकरण" in chunks[0].chunk_text
    assert chunks[1].section_number == "१"
    assert chunks[1].section_title == "संक्षिप्त नाम र प्रारम्भ"

    # Oversized दफा २ split at उपदफा boundaries; स्पष्टीकरण is a proviso
    # chunk linked to its operative subsection (PS-16).
    subsections = [c for c in chunks if c.level == "subsection"]
    assert all(c.parent_section == "२" for c in subsections)
    proviso = chunks[5]
    assert proviso.parent_section == "२"
    assert proviso.co_retrieve_parent_index == 4
    assert chunks[4].level == "subsection"
    assert "स्पष्टीकरण" in proviso.chunk_text

    # No दफा boundary is crossed mid-chunk: each chunk carries at most one
    # दफा heading, and top-level section chunks start with their own heading.
    for chunk in chunks:
        assert len(re.findall(r"\*\*[०-९]+\.", chunk.chunk_text)) <= 1
    assert chunks[6].chunk_text.startswith("**३.")

    # <amend> provenance kept in chunk_text, stripped from embed_text (PS-10).
    assert "<amend>" in chunks[1].chunk_text
    assert "<amend>" not in chunks[1].embed_text
    assert "[संशोधित]" in chunks[1].embed_text


def test_laws_chunker_real_oversized_dafa_with_provisos() -> None:
    record = _real_law_record("a0a287ee-559e-5ef9-8931-8871bf6d0c63")
    chunks = LawsChunker().chunk_text(str(record["content"]))
    section_18 = [c for c in chunks if c.section_number == "१८"]

    assert record["name"] == "सुशासन_(व्यवस्थापन_तथा_सञ्चालन)_ऐन_२०६४"
    assert [c.level for c in section_18] == [
        "subsection",
        "proviso",
        "subsection",
        "subsection",
        "subsection",
        "proviso",
        "subsection",
        "subsection",
        "subsection",
    ]
    assert all(c.parent_section == "१८" for c in section_18)
    assert all(
        c.co_retrieve_parent_index is None
        for c in section_18
        if c.level == "subsection"
    )

    provisos = [c for c in section_18 if c.level == "proviso"]
    assert [c.co_retrieve_parent_index for c in provisos] == [22, 26]
    for proviso in provisos:
        assert proviso.co_retrieve_parent_index is not None
        parent = chunks[proviso.co_retrieve_parent_index]
        assert parent.level == "subsection"
        assert parent.section_number == proviso.section_number
        assert "स्पष्टीकरण" in proviso.chunk_text


def test_laws_chunker_real_paragraph_fallback_keeps_reconstruction_order() -> None:
    record = _real_law_record("d222906c-a478-517a-9b6e-37ecba10dc1e")
    content = str(record["content"])
    chunks = LawsChunker().chunk_text(content)
    section_8 = [c for c in chunks if c.section_number == "८"]

    assert record["name"] == "लेखापरीक्षण_ऐन_२०७५"
    assert [c.level for c in section_8] == ["subsection", "subsection"]
    assert all(c.parent_section == "८" for c in section_8)
    assert all(c.co_retrieve_parent_index is None for c in section_8)
    assert "\n\n".join(c.chunk_text for c in section_8) in content
    positions = [content.index(c.chunk_text) for c in section_8]
    assert positions == sorted(positions)


def test_nkp_chunker_sections() -> None:
    chunks = NKPChunker().chunk_text(NKP_FIXTURE)
    section_types = [c.section_type for c in chunks]
    assert section_types == ["caption", "headnote", "opinion", "order", "colophon"]

    assert "सर्वोच्च अदालत" in chunks[0].chunk_text
    assert "सिद्धान्त" in chunks[1].chunk_text
    assert chunks[2].chunk_text.startswith("न्या.")
    assert chunks[3].chunk_text.startswith("१.")
    assert "२." in chunks[3].chunk_text  # numbered tail kept together
    assert chunks[4].chunk_text.startswith("इति संवत्")
    # NKP chunks have no markup to strip.
    assert all(c.embed_text == c.chunk_text for c in chunks)


def test_pii_redactor_exact_and_token_replacement() -> None:
    appellant = "काठमाडौं जिल्ला स्थायी घर भई बस्ने सुरेशकुमार शुक्ला"
    respondent = "नेपाल सरकारको नाममा दर्ता भएको जग्गा"
    text = (
        f"वादी {appellant} ले दायर गरेको मुद्दामा सुरेशकुमार शुक्ला उपस्थित भए । "
        f"प्रतिवादी {respondent} विरुद्धको फैसला ।"
    )
    redacted, warnings = PIIRedactor(enable_llm=False).redact(
        text, appellant, respondent, document_id="test-case"
    )
    assert warnings == []
    assert "[[वादी]]" in redacted
    assert "[[प्रतिवादी]]" in redacted
    assert appellant not in redacted
    assert respondent not in redacted
    # Individual name mention (not the full string) is also redacted.
    assert "सुरेशकुमार" not in redacted
    assert "शुक्ला" not in redacted


def test_pii_redactor_verification_raises_on_surviving_token() -> None:
    redactor = PIIRedactor(enable_llm=False)
    with pytest.raises(RedactionVerificationError) as exc_info:
        redactor._verify(
            "यो पाठमा सुरेशकुमार शब्द अझै छ ।",
            {"सुरेशकुमार", "शुक्ला"},
            "case-10481",
        )
    assert exc_info.value.token == "सुरेशकुमार"
    assert exc_info.value.document_id == "case-10481"


def test_langfuse_span_end_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stage span must call .end(), so endTime is not null."""
    from app.ingestion import pipeline as pipeline_mod

    ended_spans: list[str] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            self.name = name

        def end(self, output: dict[str, Any] | None = None, **kwargs: Any) -> None:
            ended_spans.append(self.name)

    class FakeTrace:
        def span(self, name: str, input: dict[str, Any] | None = None) -> FakeSpan:
            return FakeSpan(name)

        def update(self, **kwargs: Any) -> None:
            pass

        def end(self) -> None:
            pass

    def fake_trace(**kwargs: Any) -> FakeTrace:
        return FakeTrace()

    def fake_flush() -> None:
        pass

    def fake_parse_law(record: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(uri="/work/test", components=[])

    def fake_upsert_work(conn: Any, law: Any) -> str:
        return "work-id"

    def fake_authority_write(*args: Any, **kwargs: Any) -> None:
        pass

    fake_lf = SimpleNamespace(trace=fake_trace, flush=fake_flush)
    monkeypatch.setattr(pipeline_mod, "_get_lf_client", lambda: fake_lf)
    monkeypatch.setattr(pipeline_mod, "parse_law", fake_parse_law)
    monkeypatch.setattr(pipeline_mod, "upsert_work", fake_upsert_work)
    monkeypatch.setattr(pipeline_mod, "upsert_source", fake_authority_write)
    monkeypatch.setattr(pipeline_mod, "upsert_component", fake_authority_write)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", fake_authority_write)

    conn = MagicMock()
    pipeline = IngestionPipeline(conn, enable_llm=False)
    pipeline._find_existing = MagicMock(return_value=None)
    pipeline._insert_document = MagicMock(return_value="doc-id")
    pipeline._commence_date = MagicMock(return_value=None)
    pipeline._embed = MagicMock(return_value=([[0.1]], 0))
    pipeline._indexer.upsert_document = MagicMock(return_value="doc-id")

    record = {
        "_id": "law-obs",
        "name": "परीक्षण ऐन",
        "document_type": "act",
        "content": "**१. परीक्षण:** यो परीक्षण पाठ हो ।",
    }

    assert pipeline.ingest_law(record) == "doc-id"
    for stage in [
        "stage.LOAD",
        "stage.VALIDATE",
        "stage.PERSIST_AUTHORITY",
        "stage.PROPOSE_LIFECYCLE",
        "stage.CHUNK",
        "stage.EXTRACT_METADATA",
        "stage.EMBED_AND_UPSERT",
        "stage.DUAL_APPROVAL_PAUSE",
    ]:
        assert stage in ended_spans


def test_enrich_law_llm_call_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_call_count == 1 + ceil(chunk_count / CHUNK_BATCH_SIZE)."""
    import math

    def fake_call_llm(*args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
        return (
            '[{"chunk_index": 0, "keywords": [], "relevant_questions": []}]',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(metadata_enricher, "_call_llm", fake_call_llm)
    chunks = [
        LawChunk(i, f"text {i}", f"text {i}", "section", "१", None, None, None, None)
        for i in range(45)
    ]

    _, llm_calls, total_in, total_out = metadata_enricher.enrich_law_chunks(
        {"name": "परीक्षण ऐन"}, chunks
    )

    expected = 1 + math.ceil(45 / metadata_enricher.CHUNK_BATCH_SIZE)
    assert llm_calls == expected
    assert total_in == 10 * expected
    assert total_out == 5 * expected


def test_content_hash_digit_folds_and_canonicalizes_whitespace() -> None:
    assert _content_hash("दफा १\n\tपाठ") == _content_hash("दफा 1 पाठ")


def test_pipeline_idempotency_skips_unchanged_document() -> None:
    record = {
        "_id": "law-1",
        "name": "परीक्षण ऐन",
        "english_name": "Test Act",
        "document_type": "act",
        "content": "**१. परीक्षण:** यो परीक्षण पाठ हो ।",
    }
    content_hash = _content_hash(str(record["content"]))

    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ("existing-doc-id", content_hash)

    pipeline = IngestionPipeline(conn, enable_llm=False)
    pipeline._indexer = MagicMock()

    assert pipeline.ingest_law(record) is None
    assert pipeline.last_outcome == "skipped"
    pipeline._indexer.embed_chunks.assert_not_called()
    pipeline._indexer.upsert_document.assert_not_called()


def _pipeline_for_law(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[IngestionPipeline, MagicMock]:
    from app.ingestion import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "_get_lf_client", lambda: None)
    monkeypatch.setattr(pipeline_mod, "upsert_work", lambda conn, law: "work-id")
    conn = MagicMock()
    pipeline = IngestionPipeline(conn, enable_llm=False)
    pipeline._find_existing = MagicMock(return_value=None)
    pipeline._insert_document = MagicMock(return_value="doc-id")
    pipeline._commence_date = MagicMock(return_value=None)
    pipeline._embed = MagicMock(return_value=([[0.1]] * 20, 0))
    pipeline._indexer.upsert_document = MagicMock(return_value="doc-id")
    return pipeline, conn


def test_ingest_law_persists_components(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ingestion import pipeline as pipeline_mod

    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(pipeline_mod, "upsert_source", lambda *a, **k: "source-id")
    monkeypatch.setattr(
        pipeline_mod,
        "upsert_component",
        lambda conn, work_id, component: calls.append((work_id, component)),
    )
    monkeypatch.setattr(pipeline_mod, "upsert_expression", lambda *a, **k: None)
    pipeline, _ = _pipeline_for_law(monkeypatch)

    assert (
        pipeline.ingest_law(
            {"_id": "law-c", "name": "परीक्षण_ऐन_२०८०", "content": LAW_FIXTURE}
        )
        == "doc-id"
    )
    assert len(calls) >= 2
    assert {work_id for work_id, _ in calls} == {"work-id"}


def test_ingest_law_persists_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ingestion import pipeline as pipeline_mod

    source_calls: list[tuple[str, Any, Any]] = []

    def source(conn: Any, work_id: str, law: Any, source_url: Any = None) -> str:
        source_calls.append((work_id, law, source_url))
        return "source-id"

    monkeypatch.setattr(pipeline_mod, "upsert_source", source)
    monkeypatch.setattr(pipeline_mod, "upsert_component", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", lambda *a, **k: None)
    pipeline, conn = _pipeline_for_law(monkeypatch)

    pipeline.ingest_law(
        {"_id": "law-s", "name": "परीक्षण_ऐन_२०८०", "content": LAW_FIXTURE}
    )

    assert len(source_calls) == 1
    assert source_calls[0][0] == "work-id"
    assert source_calls[0][1].uri.endswith("/law-s")
    assert source_calls[0][2] is None
    conn.cursor.return_value.__enter__.return_value.execute.assert_any_call(
        "UPDATE documents SET source_pub_id = %s WHERE id = %s",
        ("source-id", "doc-id"),
    )


def test_ingest_law_sets_chunk_component_uris(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ingestion import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "upsert_source", lambda *a, **k: "source-id")
    monkeypatch.setattr(pipeline_mod, "upsert_component", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", lambda *a, **k: None)
    pipeline, _ = _pipeline_for_law(monkeypatch)

    pipeline.ingest_law(
        {"_id": "law-uri", "name": "परीक्षण_ऐन_२०८०", "content": LAW_FIXTURE}
    )

    document = pipeline._indexer.upsert_document.call_args.args[0]
    chunks = pipeline._indexer.upsert_document.call_args.args[1]
    assert document["source_pub_id"] == "source-id"
    assert [getattr(c, "component_uri", None) for c in chunks[:3]] == [
        None,
        "/np/act/2080/law-uri/dafa/1",
        "/np/act/2080/law-uri/dafa/2",
    ]


def test_ingest_law_persists_expression_with_todays_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    as_ofs: list[date] = []
    monkeypatch.setattr(pipeline_mod, "upsert_source", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "upsert_component", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline_mod,
        "upsert_expression",
        lambda conn, component, as_of: as_ofs.append(as_of),
    )
    pipeline, _ = _pipeline_for_law(monkeypatch)

    pipeline.ingest_law(
        {"_id": "law-e", "name": "परीक्षण_ऐन_२०८०", "content": LAW_FIXTURE}
    )

    assert as_ofs
    assert set(as_ofs) == {date.today()}


def test_ingest_law_skip_path_no_persistence_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    record = {"_id": "law-skip", "name": "परीक्षण ऐन", "content": "**१. परीक्षण:** पाठ ।"}
    content_hash = _content_hash(str(record["content"]))
    source = MagicMock()
    component = MagicMock()
    expression = MagicMock()
    monkeypatch.setattr(pipeline_mod, "upsert_source", source)
    monkeypatch.setattr(pipeline_mod, "upsert_component", component)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", expression)
    conn = MagicMock()
    pipeline = IngestionPipeline(conn, enable_llm=False)
    pipeline._find_existing = MagicMock(return_value=("doc-id", content_hash))

    assert pipeline.ingest_law(record) is None
    source.assert_not_called()
    component.assert_not_called()
    expression.assert_not_called()


def test_ingest_law_persistence_failure_rejects_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "upsert_source", lambda *a, **k: None)

    def fail_component(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_mod, "upsert_component", fail_component)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", lambda *a, **k: None)
    span_outputs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(pipeline_mod, "_begin_span", lambda trace, stage, input: stage)
    monkeypatch.setattr(
        pipeline_mod,
        "_end_span",
        lambda span, output: span_outputs.append((span, output)),
    )
    pipeline, conn = _pipeline_for_law(monkeypatch)
    pipeline._set_status = MagicMock()
    pipeline._laws_chunker.chunk_text = MagicMock(return_value=[])

    assert (
        pipeline.ingest_law(
            {"_id": "law-f", "name": "परीक्षण_ऐन_२०८०", "content": LAW_FIXTURE}
        )
        is None
    )
    pipeline._set_status.assert_called_once_with("doc-id", "rejected")
    assert pipeline.last_outcome == "rejected"
    conn.commit.assert_called_once()
    pipeline._laws_chunker.chunk_text.assert_not_called()
    persist_output = dict(span_outputs)["PERSIST_AUTHORITY"]
    assert "error" not in persist_output
    assert persist_output["error_type"] == "RuntimeError"


def test_ingest_law_rejects_duplicate_component_uris(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    source = MagicMock()
    component = MagicMock()
    expression = MagicMock()
    law = SimpleNamespace(
        uri="/np/act/2080/dup",
        components=[SimpleNamespace(uri="/u/1"), SimpleNamespace(uri="/u/1")],
    )
    monkeypatch.setattr(pipeline_mod, "parse_law", lambda record: law)
    monkeypatch.setattr(pipeline_mod, "upsert_source", source)
    monkeypatch.setattr(pipeline_mod, "upsert_component", component)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", expression)
    pipeline, _ = _pipeline_for_law(monkeypatch)

    assert (
        pipeline.ingest_law(
            {"_id": "dup-law", "name": "खराब_ऐन_२०८०", "content": "**१. दफा:** पाठ"}
        )
        is None
    )
    assert pipeline.last_outcome == "rejected"
    source.assert_not_called()
    component.assert_not_called()
    expression.assert_not_called()


def test_ingest_law_validate_failure_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    source = MagicMock()
    component = MagicMock()
    expression = MagicMock()
    monkeypatch.setattr(pipeline_mod, "upsert_source", source)
    monkeypatch.setattr(pipeline_mod, "upsert_component", component)
    monkeypatch.setattr(pipeline_mod, "upsert_expression", expression)
    pipeline, _ = _pipeline_for_law(monkeypatch)

    assert (
        pipeline.ingest_law(
            {"_id": "bad-law", "name": "खराब_ऐन_२०८०", "content": "दफा छैन"}
        )
        is None
    )
    source.assert_not_called()
    component.assert_not_called()
    expression.assert_not_called()


def _check_constraint_holds(
    status: str, approved_by: str | None, second_approved_by: str | None
) -> bool:
    """Python mirror of the documents_dual_approval CHECK constraint."""
    return status != "approved" or (
        approved_by is not None
        and second_approved_by is not None
        and approved_by != second_approved_by
    )


def test_dual_approval_check_constraint_logic() -> None:
    sql = MIGRATION_005.read_text(encoding="utf-8")
    assert "documents_dual_approval" in sql
    assert "approved_by <> second_approved_by" in sql
    # Single-approver and self-approval must be rejected (PS-2).
    assert not _check_constraint_holds("approved", "gatekeeper", None)
    assert not _check_constraint_holds("approved", "alice", "alice")
    assert not _check_constraint_holds("approved", None, None)
    # Two distinct approvers pass; non-approved states are unconstrained.
    assert _check_constraint_holds("approved", "alice", "bob")
    assert _check_constraint_holds("pending", None, None)
    assert _check_constraint_holds("rejected", None, None)


@pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="no local Postgres (SUPABASE_DB_URL unset) — CHECK constraint "
    "verified by logic mirror in test_dual_approval_check_constraint_logic",
)
def test_dual_approval_check_constraint_live_db() -> None:
    import psycopg2
    import psycopg2.errors

    sql = MIGRATION_005.read_text(encoding="utf-8")
    match = re.search(
        r"CONSTRAINT documents_dual_approval CHECK \((.*?)\)\s*\)", sql, re.DOTALL
    )
    assert match, "dual-approval CHECK not found in migration 005"
    check_expr = match.group(1)

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=3)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE dual_approval_probe ("
                "ingestion_status text, approved_by text, second_approved_by text, "
                f"CONSTRAINT documents_dual_approval CHECK ({check_expr}))"
            )
            cur.execute(
                "INSERT INTO dual_approval_probe VALUES ('pending', NULL, NULL)"
            )
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO dual_approval_probe VALUES "
                    "('approved', 'alice', NULL)"
                )
            conn.rollback()
            with conn.cursor() as cur2:
                cur2.execute(
                    "CREATE TEMP TABLE dual_approval_probe2 ("
                    "ingestion_status text, approved_by text, "
                    "second_approved_by text, "
                    f"CONSTRAINT documents_dual_approval CHECK ({check_expr}))"
                )
                with pytest.raises(psycopg2.errors.CheckViolation):
                    cur2.execute(
                        "INSERT INTO dual_approval_probe2 VALUES "
                        "('approved', 'alice', 'alice')"
                    )
    finally:
        conn.rollback()
        conn.close()
