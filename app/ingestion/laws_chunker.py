"""
Structure-aware chunker for Nepali acts (laws.jsonl).

Design: docs/ingestion_design.md §2.1. Splits on दफा anchors (``**N.`` bold
headings), sub-splits oversized दफा at उपदफा boundaries, and keeps
स्पष्टीकरण/proviso blocks co-retrievable with their operative clause (PS-16).
No fixed-size chunking with overlap anywhere.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ~1,800 chars target, 2,400 hard max, 400 soft min (design §2).
MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 400  # soft floor; legal units are never merged (see task spec)

# दफा heading anchor: **१. Title:** (bold, Devanagari numeral).
_DAFA_HEADING_RE = re.compile(r"\*\*([०-९]+)\.\s*([^\n*]+?)\*\*")
_DAFA_SPLIT_RE = re.compile(r"(?=\*\*[०-९]+\.)")
# Chapter header: ## परिच्छेद-१
_CHAPTER_RE = re.compile(r"##\s*परिच्छेद[-–]\s*([०-९]+)")
# उपदफा boundary at line start: (१) (२) …
_SUBSECTION_SPLIT_RE = re.compile(r"(?m)^(?=\([०-९]+\))")
# स्पष्टीकरण block opener.
_PROVISO_RE = re.compile(r"(?m)^\s*स्पष्टीकरण\s*[:：]")
# Provenance markup stripped from embed_text only (chunk_text keeps it, PS-10).
_AMEND_RE = re.compile(r"<amend>[^<]+</amend>")
_ELISION_RE = re.compile(r"✂+\.+")
# Fallback boundaries when a single उपदफा still exceeds the max (design §2.1:
# split at paragraph boundaries with parent_section retained).
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=।)\s+")


@dataclass
class LawChunk:
    chunk_index: int
    chunk_text: str  # verbatim from content (including <amend> tags)
    embed_text: str  # <amend>…</amend> and ✂ stripped, for embedding only
    level: str  # 'act' | 'chapter' | 'section' | 'subsection' | 'proviso'
    section_number: str | None
    section_title: str | None
    chapter_number: str | None
    parent_section: str | None  # दफा number when level='subsection'/'proviso'
    co_retrieve_parent_index: int | None  # chunk_index of operative clause (PS-16)
    # Denormalized cache of the authority store's approved commence effect;
    # NULL-pending when commencement is unverified (PS-2). Filled by the pipeline.
    effective_date_ad: object | None = field(default=None)
    # LLM-extracted metadata, attached by the pipeline before upsert.
    keywords: list[str] | None = field(default=None)
    relevant_questions: list[str] | None = field(default=None)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _embed_text(chunk_text: str) -> str:
    text = _AMEND_RE.sub("[संशोधित]", chunk_text)
    return _ELISION_RE.sub("[…]", text)


def _split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text that exceeds max_chars without crossing word boundaries.

    Order: paragraph boundaries, then Nepali sentence boundaries (danda),
    then last-resort whitespace. Never splits mid-word.
    """
    if len(text) <= max_chars:
        return [text]
    paragraphs = _PARAGRAPH_RE.split(text)
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(para) <= max_chars:
            current = para
            continue
        # Single paragraph still too long: split at danda sentence boundaries.
        sentences = _SENTENCE_RE.split(para)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}" if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                pieces.append(current)
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                pieces.append(sentence[:cut])
                sentence = sentence[cut:].lstrip()
            current = sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


class LawsChunker:
    """Chunks one laws.jsonl record into LawChunks (design §2.1)."""

    def chunk(self, record: dict) -> list[LawChunk]:
        content = str(record.get("content") or "")
        return self.chunk_text(content)

    def chunk_text(self, content: str) -> list[LawChunk]:
        chunks: list[LawChunk] = []

        # Chapter header positions, for chapter_number tracking.
        chapters = [(m.start(), m.group(1)) for m in _CHAPTER_RE.finditer(content)]

        def chapter_at(pos: int) -> str | None:
            enclosing = [num for start, num in chapters if start < pos]
            return enclosing[-1] if enclosing else None

        # 1. Preamble/header block before the first दफा heading → level='act'.
        first_heading = _DAFA_SPLIT_RE.search(content)
        preamble = content[: first_heading.start()] if first_heading else content
        if preamble.strip():
            self._emit(chunks, preamble, "act", None, None, None, None, None)
        if not first_heading:
            return chunks

        # 2. Split on दफा boundaries, tracking each block's offset so chapter
        # attribution is exact even when two दफा blocks share identical text.
        heading_starts = [m.start() for m in _DAFA_SPLIT_RE.finditer(content)]
        heading_starts.append(len(content))
        dafa_blocks = [
            (start, content[start:end])
            for start, end in zip(heading_starts, heading_starts[1:])
            if content[start:end].strip()
        ]
        for block_start, block in dafa_blocks:
            heading = _DAFA_HEADING_RE.match(block)
            section_number = heading.group(1) if heading else None
            section_title = (
                heading.group(2).strip().rstrip(":").strip() or None if heading else None
            )
            chapter_number = chapter_at(block_start)

            if len(block) <= MAX_CHUNK_CHARS:
                # Common case: दफा (and any inline स्पष्टीकरण) fits in one chunk.
                self._emit(
                    chunks,
                    block,
                    "section",
                    section_number,
                    section_title,
                    chapter_number,
                    None,
                    None,
                )
                continue

            # 3. Oversized दफा: split at उपदफा line-start boundaries.
            pieces = [p for p in _SUBSECTION_SPLIT_RE.split(block) if p.strip()]
            if len(pieces) <= 1:
                # No उपदफा anchors (or one giant उपदफा): paragraph fallback.
                pieces = _split_oversized(block)
            for piece in pieces:
                if len(piece) > MAX_CHUNK_CHARS:
                    for sub in _split_oversized(piece):
                        self._emit_subsection(
                            chunks, sub, section_number, section_title, chapter_number
                        )
                    continue
                self._emit_subsection(
                    chunks, piece, section_number, section_title, chapter_number
                )
        return chunks

    def _emit_subsection(
        self,
        chunks: list[LawChunk],
        piece: str,
        section_number: str | None,
        section_title: str | None,
        chapter_number: str | None,
    ) -> None:
        # 4. स्पष्टीकरण inside a split दफा becomes a 'proviso' chunk linked to
        # its operative subsection (PS-16); never a standalone top-level chunk.
        proviso = _PROVISO_RE.search(piece)
        if not proviso:
            self._emit(
                chunks,
                piece,
                "subsection",
                section_number,
                section_title,
                chapter_number,
                section_number,
                None,
            )
            return
        operative = piece[: proviso.start()]
        proviso_text = piece[proviso.start() :]
        parent_index: int | None = None
        if operative.strip():
            self._emit(
                chunks,
                operative,
                "subsection",
                section_number,
                section_title,
                chapter_number,
                section_number,
                None,
            )
            parent_index = chunks[-1].chunk_index
        elif chunks:
            # Proviso opens the piece: link to the previously emitted chunk of
            # the same दफा (its operative clause precedes it in document order).
            parent_index = chunks[-1].chunk_index
        self._emit(
            chunks,
            proviso_text,
            "proviso",
            section_number,
            section_title,
            chapter_number,
            section_number,
            parent_index,
        )

    def _emit(
        self,
        chunks: list[LawChunk],
        text: str,
        level: str,
        section_number: str | None,
        section_title: str | None,
        chapter_number: str | None,
        parent_section: str | None,
        co_retrieve_parent_index: int | None,
    ) -> None:
        # 5. NFC-normalize; 6. strip provenance markup from embed_text only;
        # 7. document order, 0-based chunk_index.
        chunk_text = _nfc(text.strip())
        chunks.append(
            LawChunk(
                chunk_index=len(chunks),
                chunk_text=chunk_text,
                embed_text=_embed_text(chunk_text),
                level=level,
                section_number=section_number,
                section_title=section_title,
                chapter_number=chapter_number,
                parent_section=parent_section,
                co_retrieve_parent_index=co_retrieve_parent_index,
            )
        )
