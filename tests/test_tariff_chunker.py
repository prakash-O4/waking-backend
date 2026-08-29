"""
Unit tests for the tariff schedule chunker and pipeline routing gate.

No LLMs or databases are touched. Deterministic metadata is verified directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.tariff_chunker import TariffChunker, is_tariff_dominant

TARIFF_TABLE = """
## भाग १०
धान, गहुँ र मकै

| शीर्षक | उपशीर्षक | वस्तुको विवरण | निकासी भन्सार महसुल दर |
|---|---|---|---|
| १ | २ | ३ | ४ |
| १०.०१ | | गहुँ र मेसलिन । | |
| | | -दुरम गहुँ: | |
| | १००१.११.०० | --बिउ | प्रति कि.ग्रा. रु.१।- |
| | १००१.१९.०० | --अन्य | प्रति कि.ग्रा. रु.१।- |
| १०.०५ | | मकै । | |
| | १००५.१०.०० | -बिउ | प्रति कि.ग्रा. रु.१।- |
| | १००५.९०.०० | -अन्य | प्रति कि.ग्रा. रु.१।- |
| २३.०४ | २३०४.००.०० | भटमासको तेल निकाल्दा प्राप्त हुने पिना र अन्य ठोस अवशेष। | प्रति कि.ग्रा. रु.५०।- |
"""

PROSE_CONTENT = """
## परिच्छेद-१
प्रारम्भिक

**१. संक्षिप्त नाम र प्रारम्भ:** (१) यस ऐनको नाम "परीक्षण ऐन, २०८०" रहेको छ ।

**२. परिभाषा:** विषय वा प्रसङ्गले अर्को अर्थ नलागेमा यस ऐनमा,-

(क) "करमुक्त पसल" भन्नाले बैङ्क जमानत सुविधामा पैठारी गरिएका मालवस्तु कूटनीतिक सुविधा वा महसुल सुविधा प्राप्त व्यक्तिलाई बिक्री गर्ने पसल सम्झनु पर्छ ।
"""

PIPE_ONLY_CONTENT = """
| सि.नं. | विवरण | मात्रा |
|---|---|---|
| १ | भवन निर्माण सामग्री | १० |
| २ | इन्जिनियरिङ उपकरण | ५ |
| ३ | कार्यालय फर्निचर | १२ |
"""


def test_is_tariff_dominant_true() -> None:
    # Build content with >5000 HS-code-like patterns plus a tariff keyword.
    rows = "| १०.०१ | | गहुँ र मेसलिन |\n"
    rows += "\n".join(
        f"| | {1000 + i:04d}.11.00 | --वस्तु {i} | प्रति कि.ग्रा. रु.१।-"
        for i in range(5100)
    )
    content = f"पैठारी महसुल दर तालिका\n\n{rows}"
    assert is_tariff_dominant(content) is True


def test_is_tariff_dominant_false_prose() -> None:
    assert is_tariff_dominant(PROSE_CONTENT) is False


def test_is_tariff_dominant_false_pipe_only() -> None:
    # Many pipes but zero HS-code-like patterns and no tariff keyword.
    assert "|" in PIPE_ONLY_CONTENT
    assert is_tariff_dominant(PIPE_ONLY_CONTENT) is False


def test_tariff_chunker_heading_row_linkage() -> None:
    chunks = TariffChunker().chunk_text(TARIFF_TABLE, act_name="भन्सार महसुल ऐन २०८१")

    headings = [c for c in chunks if c.level == "tariff_heading"]
    rows = [c for c in chunks if c.level == "tariff_row"]
    assert headings
    assert rows

    for heading in headings:
        children = [
            c for c in rows if c.co_retrieve_parent_index == heading.chunk_index
        ]
        assert children, f"heading {heading.section_number} has no linked rows"
        for child in children:
            assert child.parent_section == heading.section_number


def test_tariff_chunker_embed_text_context_rich() -> None:
    chunks = TariffChunker().chunk_text(TARIFF_TABLE, act_name="भन्सार महसुल ऐन २०८१")

    row = next(c for c in chunks if c.section_number == "१००१.११.००")
    assert "भन्सार महसुल ऐन २०८१" in row.embed_text
    assert "१००१.११.००" in row.embed_text
    assert "--बिउ" in row.embed_text or "बिउ" in row.embed_text
    assert "|" not in row.embed_text
    assert "महसुल दर" in row.embed_text

    heading = next(c for c in chunks if c.section_number == "१०.०१")
    assert "शीर्षक १०.०१" in heading.embed_text
    assert "गहुँ र मेसलिन" in heading.embed_text
    assert "|" not in heading.embed_text


def test_tariff_chunker_deterministic_questions() -> None:
    chunks = TariffChunker().chunk_text(TARIFF_TABLE, act_name="भन्सार महसुल ऐन २०८१")

    row = next(c for c in chunks if c.section_number == "१००१.११.००")
    assert row.keywords
    assert row.relevant_questions
    assert any("महसुल दर" in q for q in row.relevant_questions)
    assert any("१००१.११.००" in q for q in row.relevant_questions)
    assert any("बिउ" in k for k in row.keywords)

    heading = next(c for c in chunks if c.section_number == "१०.०१")
    assert heading.keywords
    assert heading.relevant_questions
    assert any("१०.०१" in q for q in heading.relevant_questions)
    assert any("गहुँ र मेसलिन" in k for k in heading.keywords)


def test_pipeline_routes_tariff_to_tariff_chunker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ingestion import pipeline as pipeline_mod

    # Include a दफा anchor so the validation gate passes; routing happens later.
    tariff_content = "**१. परीक्षण:** यो परीक्षण पाठ हो ।\n| १०.०१ | | गहुँ |\n| | १००१.११.०० | बिउ | रु.१ |\n"
    record = {
        "_id": "tariff-1",
        "name": "भन्सार महसुल ऐन २०८१",
        "document_type": "act",
        "content": tariff_content,
    }

    conn = MagicMock()
    pipeline = IngestionPipeline(conn, enable_llm=True)
    pipeline._find_existing = MagicMock(return_value=None)
    pipeline._insert_document = MagicMock(return_value="doc-id")
    pipeline._commence_date = MagicMock(return_value=None)
    pipeline._embed = MagicMock(return_value=([[0.1]], 0))
    pipeline._indexer.upsert_document = MagicMock(return_value="doc-id")

    with patch.object(
        pipeline_mod, "is_tariff_dominant", return_value=True
    ) as mock_is_tariff, patch.object(
        pipeline._tariff_chunker, "chunk_text", return_value=[]
    ) as mock_tariff_chunk, patch.object(
        pipeline._laws_chunker, "chunk_text"
    ) as mock_laws_chunk, patch.object(
        pipeline_mod.metadata_enricher, "enrich_law_chunks"
    ) as mock_enrich:
        # chunk_text returns [] so the pipeline rejects; we only care about routing.
        assert pipeline.ingest_law(record) is None

    mock_is_tariff.assert_called_once_with(tariff_content)
    mock_tariff_chunk.assert_called_once_with(
        tariff_content, act_name="भन्सार महसुल ऐन २०८१"
    )
    mock_laws_chunk.assert_not_called()
    mock_enrich.assert_not_called()
