"""
Structure-aware chunker for tariff schedule acts (e.g. भन्सार_महसुल_ऐन_२०८१).

Parses pipe-table rows into heading/subheading/note chunks with deterministic
metadata and parent-child linkage (PS-16). No LLM is used.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Detection signals used by the pipeline routing gate.
_HS_CODE_RE = re.compile(r"[०-९\d]{2,4}\.[०-९\d]{2}")
_TARIFF_KW_RE = re.compile(r"(सार्क|पैठारी|उपशीर्षक|महसुल दर)")

# Row parsing patterns.
_HEADING_HS_RE = re.compile(r"^[०-९\d]{2}\.[०-९\d]{2}$")
_SUBHEADING_HS_RE = re.compile(r"^[०-९\d]{4}\.[०-९\d]{2}\.[०-९\d]{2}$")
_CHAPTER_RE = re.compile(r"##?\s*भाग[-–\s]*([०-९]+)")

# Header rows to ignore in each table block.
_HEADER_LABELS = {
    "शीर्षक",
    "उपशीर्षक",
    "वस्तुको विवरण",
    "निकासी भन्सार महसुल दर",
    "पैठारी महसुल दर",
    "सार्क मुलुकबाट",
    "अन्य मुलुकबाट",
}


def is_tariff_dominant(content: str) -> bool:
    """True only for acts that are dominated by HS-code tariff tables."""
    return len(_HS_CODE_RE.findall(content)) > 5000 and bool(
        _TARIFF_KW_RE.search(content)
    )


@dataclass
class TariffChunk:
    chunk_index: int
    chunk_text: str  # verbatim pipe-table row(s), NFC-normalized
    embed_text: str  # context-rich structured text for embedding
    level: str  # 'tariff_heading' | 'tariff_row' | 'tariff_note'
    section_number: str | None  # HS code of this row
    section_title: str | None  # goods description / heading title
    chapter_number: str | None  # भाग/part number if detectable
    parent_section: str | None  # heading HS code when level='tariff_row'
    co_retrieve_parent_index: int | None  # chunk_index of parent heading (PS-16)
    effective_date_ad: object | None = field(default=None)
    keywords: list[str] | None = field(default=None)
    relevant_questions: list[str] | None = field(default=None)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _strip_html_breaks(text: str) -> str:
    return re.sub(r"\s*<br\s*/?>\s*", " ", text).strip()


def _clean_cell(text: str) -> str:
    text = _nfc(text.strip())
    text = _strip_html_breaks(text)
    return text


def _is_header_row(cells: list[str]) -> bool:
    """True for table header / column-number rows that should not become chunks."""
    if not cells:
        return True
    if cells[0] in {"१", "1"} and len(cells) >= 4 and cells[1] in {"२", "2"}:
        return True
    if any(cell in _HEADER_LABELS for cell in cells):
        return True
    return False


def _duty_rate_text(duty_cells: list[str]) -> str | None:
    parts = [_clean_cell(c) for c in duty_cells if _clean_cell(c)]
    return " ".join(parts) or None


def _strip_danda(text: str) -> str:
    return text.rstrip("। ")


def _embed_text(
    act_name: str | None,
    chapter_number: str | None,
    chapter_title: str | None,
    heading_code: str | None,
    heading_description: str | None,
    subheading_code: str | None,
    goods_description: str | None,
    duty_rate: str | None,
    level: str,
) -> str:
    lines: list[str] = []
    if act_name:
        lines.append(f"ऐन: {_strip_danda(act_name)}।")
    if chapter_number:
        if chapter_title:
            lines.append(f"भाग {chapter_number}: {_strip_danda(chapter_title)}।")
        else:
            lines.append(f"भाग {chapter_number}।")
    if heading_code:
        if heading_description:
            lines.append(f"शीर्षक {heading_code}: {_strip_danda(heading_description)}।")
        else:
            lines.append(f"शीर्षक {heading_code}।")
    if subheading_code:
        lines.append(f"उपशीर्षक {subheading_code}।")
    if goods_description:
        lines.append(f"वस्तु: {_strip_danda(goods_description)}।")
    if duty_rate and level == "tariff_row":
        lines.append(f"महसुल दर: {_strip_danda(duty_rate)}।")
    if level == "tariff_note" and goods_description:
        lines.append(f"नोट: {_strip_danda(goods_description)}।")
    return "\n".join(lines)


def _keywords_and_questions(
    level: str,
    act_name: str | None,
    chapter_title: str | None,
    heading_code: str | None,
    heading_description: str | None,
    subheading_code: str | None,
    goods_description: str | None,
) -> tuple[list[str], list[str]]:
    if level == "tariff_row":
        keywords = [
            x
            for x in [
                goods_description,
                heading_code,
                subheading_code,
                chapter_title,
                act_name,
            ]
            if x
        ]
        questions = [
            f"{goods_description} को महसुल दर कति हो?",
            f"{subheading_code} अन्तर्गत कुन वस्तु पर्छ?",
            f"{goods_description} पैठारी गर्दा महसुल कति लाग्छ?",
        ]
    elif level == "tariff_heading":
        keywords = [
            x for x in [heading_description, heading_code, chapter_title, act_name] if x
        ]
        questions = [
            f"{heading_description} को भन्सार शीर्षक कोड के हो?",
            f"शीर्षक {heading_code} अन्तर्गत कुन वस्तुहरू पर्छन्?",
        ]
    else:  # tariff_note
        keywords = [
            x for x in [goods_description, heading_code, chapter_title, act_name] if x
        ]
        questions = []
    return keywords, questions


class TariffChunker:
    """Chunks one tariff schedule act into row-level TariffChunks."""

    def chunk(self, record: dict[str, Any]) -> list[TariffChunk]:
        content = str(record.get("content") or "")
        return self.chunk_text(content, act_name=record.get("name", ""))

    def chunk_text(self, content: str, act_name: str = "") -> list[TariffChunk]:
        content = _nfc(content)
        chunks: list[TariffChunk] = []

        # Detect chapter headers and their titles.
        chapter_positions: list[tuple[int, str, str | None]] = []
        for m in _CHAPTER_RE.finditer(content):
            chapter_positions.append((m.start(), m.group(1), None))
        # Also capture nearby heading text (line after the chapter marker).
        lines = content.split("\n")
        line_starts = [0]
        for line in lines[:-1]:
            line_starts.append(line_starts[-1] + len(line) + 1)

        def chapter_at(pos: int) -> tuple[str | None, str | None]:
            enclosing = [
                (num, title) for start, num, title in chapter_positions if start < pos
            ]
            return enclosing[-1] if enclosing else (None, None)

        # State tracked across table blocks.
        current_heading_code: str | None = None
        current_heading_description: str | None = None
        current_heading_index: int | None = None
        current_chapter_number: str | None = None
        current_chapter_title: str | None = None

        # Group consecutive pipe lines into table blocks.
        table_blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in lines:
            if line.strip().startswith("|"):
                current_block.append(line)
            else:
                if current_block:
                    table_blocks.append(current_block)
                    current_block = []
        if current_block:
            table_blocks.append(current_block)

        for block in table_blocks:
            for raw_line in block:
                cells = [_clean_cell(c) for c in raw_line.split("|")]
                # Remove only the single outermost empty cell introduced by each
                # end pipe; internal empty columns must be preserved.
                if cells and not cells[0]:
                    cells.pop(0)
                if cells and not cells[-1]:
                    cells.pop()

                if _is_header_row(cells):
                    continue

                line_start = content.find(raw_line)
                chapter_number, _ = chapter_at(line_start)
                if chapter_number:
                    current_chapter_number = chapter_number

                heading_cell = cells[0] if cells else ""
                subheading_cell = cells[1] if len(cells) > 1 else ""
                description_cell = cells[2] if len(cells) > 2 else ""
                duty_cells = cells[3:] if len(cells) > 3 else []

                is_heading = bool(_HEADING_HS_RE.match(heading_cell))
                is_subheading = bool(_SUBHEADING_HS_RE.match(subheading_cell))

                # Combined heading + subheading row (heading has only one subheading).
                if is_heading and is_subheading:
                    # Emit the heading first.
                    self._emit(
                        chunks,
                        raw_line,
                        act_name,
                        current_chapter_number,
                        current_chapter_title,
                        level="tariff_heading",
                        heading_code=heading_cell,
                        heading_description=description_cell,
                        subheading_code=None,
                        goods_description=description_cell,
                        duty_rate=None,
                        parent_section=None,
                        parent_index=None,
                    )
                    current_heading_code = heading_cell
                    current_heading_description = description_cell
                    current_heading_index = chunks[-1].chunk_index
                    # Then emit the subheading row linked to that heading.
                    duty_rate = _duty_rate_text(duty_cells)
                    self._emit(
                        chunks,
                        raw_line,
                        act_name,
                        current_chapter_number,
                        current_chapter_title,
                        level="tariff_row",
                        heading_code=current_heading_code,
                        heading_description=current_heading_description,
                        subheading_code=subheading_cell,
                        goods_description=description_cell,
                        duty_rate=duty_rate,
                        parent_section=current_heading_code,
                        parent_index=current_heading_index,
                    )
                    continue

                if is_heading:
                    current_heading_code = heading_cell
                    current_heading_description = description_cell
                    self._emit(
                        chunks,
                        raw_line,
                        act_name,
                        current_chapter_number,
                        current_chapter_title,
                        level="tariff_heading",
                        heading_code=heading_cell,
                        heading_description=description_cell,
                        subheading_code=None,
                        goods_description=description_cell,
                        duty_rate=None,
                        parent_section=None,
                        parent_index=None,
                    )
                    current_heading_index = chunks[-1].chunk_index
                    continue

                if is_subheading:
                    duty_rate = _duty_rate_text(duty_cells)
                    self._emit(
                        chunks,
                        raw_line,
                        act_name,
                        current_chapter_number,
                        current_chapter_title,
                        level="tariff_row",
                        heading_code=current_heading_code,
                        heading_description=current_heading_description,
                        subheading_code=subheading_cell,
                        goods_description=description_cell,
                        duty_rate=duty_rate,
                        parent_section=current_heading_code,
                        parent_index=current_heading_index,
                    )
                    continue

                # Note / subtotal / intermediate grouping row.
                if description_cell and (
                    current_heading_code or current_heading_index is not None
                ):
                    self._emit(
                        chunks,
                        raw_line,
                        act_name,
                        current_chapter_number,
                        current_chapter_title,
                        level="tariff_note",
                        heading_code=current_heading_code,
                        heading_description=current_heading_description,
                        subheading_code=None,
                        goods_description=description_cell,
                        duty_rate=None,
                        parent_section=current_heading_code,
                        parent_index=current_heading_index,
                        section_number=None,
                    )

        return chunks

    def _emit(
        self,
        chunks: list[TariffChunk],
        raw_line: str,
        act_name: str,
        chapter_number: str | None,
        chapter_title: str | None,
        level: str,
        heading_code: str | None,
        heading_description: str | None,
        subheading_code: str | None,
        goods_description: str | None,
        duty_rate: str | None,
        parent_section: str | None,
        parent_index: int | None,
        section_number: str | None = None,
    ) -> None:
        chunk_text = _nfc(raw_line.strip())
        if section_number is None:
            section_number = subheading_code or heading_code
        section_title = goods_description or heading_description

        embed_text = _embed_text(
            act_name=act_name or None,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            heading_code=heading_code,
            heading_description=heading_description,
            subheading_code=subheading_code,
            goods_description=goods_description,
            duty_rate=duty_rate,
            level=level,
        )

        keywords, relevant_questions = _keywords_and_questions(
            level=level,
            act_name=act_name or None,
            chapter_title=chapter_title,
            heading_code=heading_code,
            heading_description=heading_description,
            subheading_code=subheading_code,
            goods_description=goods_description,
        )

        chunks.append(
            TariffChunk(
                chunk_index=len(chunks),
                chunk_text=chunk_text,
                embed_text=embed_text,
                level=level,
                section_number=section_number,
                section_title=section_title,
                chapter_number=chapter_number,
                parent_section=parent_section,
                co_retrieve_parent_index=parent_index,
                keywords=keywords,
                relevant_questions=relevant_questions,
            )
        )
