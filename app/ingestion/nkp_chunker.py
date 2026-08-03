"""
Hybrid chunker for NKP Supreme Court decisions (output/nkp_cases.jsonl).

Design: docs/ingestion_design.md §2.2. Primary split on deterministic anchors
(caption / headnote / advocates / opinion / order / colophon), size-driven
secondary split for oversized chunks. Anything that fails to match falls into
section_type='body' — it is never force-fit and never silently dropped.

Redaction happens BEFORE chunking in the pipeline, so this chunker only ever
sees redacted text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ~1,800 chars target, 2,400 hard max (design §2).
MAX_CHUNK_CHARS = 2400

# Primary anchors (§2.2 table, applied in order).
_CAPTION_LINE_RES = [
    re.compile(r"^सर्वोच्च अदालत"),
    re.compile(r"^माननीय न्यायाधीश"),
    re.compile(r"^आदेश मिति\s*:"),
    re.compile(r"^मुद्दाः"),
    re.compile(r"^विषयः"),
]
_ADVOCATE_LINE_RE = re.compile(r"(?m)^[^।\n]*का तर्फबाट\s*:")
# Note: task.md gives the opener as `न्या\.[^\s]+\s*:`, but real NKP data has
# full judge names with spaces before the colon (`न्या.तिलप्रसाद श्रेष्ठ :`),
# which that pattern cannot match; this form matches the observed opener.
_OPINION_RE = re.compile(r"(?m)^न्या\.[^:\n]*:")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^([०-९]+)\.")
_COLOPHON_RE = re.compile(r"(?m)^इति संवत्")
_SENTENCE_RE = re.compile(r"(?<=।)\s+")

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

SECTION_LABELS = {
    "caption": "इजलास विवरण",
    "headnote": "सिद्धान्त सार",
    "advocates": "अधिवक्ता बहस",
    "opinion": "न्यायाधीशको राय",
    "order": "आदेश",
    "colophon": "इति संवत्",
    "body": "मुख्य पाठ",
}


@dataclass
class NKPChunk:
    chunk_index: int
    chunk_text: str  # redacted text
    embed_text: str  # same as chunk_text (NKP has no markup to strip)
    section_type: str  # caption|headnote|advocates|opinion|order|colophon|body
    section_label: str  # human-readable label
    # LLM-extracted metadata, attached by the pipeline before upsert.
    keywords: list[str] | None = field(default=None)
    relevant_questions: list[str] | None = field(default=None)


def _devanagari_int(numeral: str) -> int | None:
    try:
        return int(numeral.translate(_DEVANAGARI_DIGITS))
    except ValueError:
        return None


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    """Greedily pack segments into pieces ≤ max_chars; never split mid-word."""
    pieces: list[str] = []
    current = ""
    for segment in sentences:
        candidate = f"{current} {segment}" if current else segment
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(segment) > max_chars:
            cut = segment.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(segment[:cut])
            segment = segment[cut:].lstrip()
        current = segment
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def _secondary_split(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Size-driven fallback split (§2.2): numbered-item boundaries first, then
    danda sentence boundaries. Never mid-word."""
    if len(text) <= max_chars:
        return [text]
    # 1. Line-start numbered-item boundaries.
    starts = [m.start() for m in _NUMBERED_ITEM_RE.finditer(text)]
    if starts:
        starts.append(len(text))
        segments = [
            text[s:e] for s, e in zip(starts, starts[1:]) if text[s:e].strip()
        ]
        # Text before the first numbered item belongs to the first segment.
        prefix = text[: starts[0]]
        if prefix.strip():
            segments[0] = prefix + segments[0]
    else:
        segments = [text]
    pieces: list[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            pieces.append(segment)
        else:
            # 2. Sentence boundary on Nepali danda.
            pieces.extend(_pack(_SENTENCE_RE.split(segment), max_chars))
    # Repack small numbered pieces up to the size target.
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return [p for p in merged if p.strip()]


class NKPChunker:
    """Chunks one redacted NKP case full_text into NKPChunks (design §2.2)."""

    def chunk(self, record: dict) -> list[NKPChunk]:
        return self.chunk_text(str(record.get("full_text") or ""))

    def chunk_text(self, text: str) -> list[NKPChunk]:
        chunks: list[NKPChunk] = []
        pos = 0

        # 1–2. Caption block: सर्वोच्च अदालत + justice lines, plus optional
        # आदेश मिति / मुद्दाः / विषयः lines when adjacent.
        caption_end = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or any(p.match(stripped) for p in _CAPTION_LINE_RES):
                caption_end += len(line)
                continue
            break
        if text[:caption_end].strip():
            self._emit(chunks, text[:caption_end], "caption")
        pos = caption_end

        # 3. Headnote: text before the first advocate line or न्या. opener.
        advocate_first = _ADVOCATE_LINE_RE.search(text, pos)
        opinion_first = _OPINION_RE.search(text, pos)
        headnote_end = len(text)
        for match in (advocate_first, opinion_first):
            if match:
                headnote_end = min(headnote_end, match.start())
        if text[pos:headnote_end].strip():
            self._emit(chunks, text[pos:headnote_end], "headnote")
        pos = headnote_end

        # 4. Advocates: consecutive advocate lines as one chunk.
        if advocate_first and advocate_first.start() == pos:
            advocate_lines: list[str] = []
            adv_pos = pos
            while True:
                match = _ADVOCATE_LINE_RE.match(text, adv_pos)
                if not match:
                    break
                line_end = text.find("\n", match.start())
                line_end = len(text) if line_end == -1 else line_end + 1
                advocate_lines.append(text[match.start() : line_end])
                adv_pos = line_end
                # Skip blank lines between advocate lines.
                while adv_pos < len(text) and text[adv_pos] in " \t\n":
                    adv_pos += 1
            if advocate_lines:
                self._emit(chunks, "".join(advocate_lines), "advocates")
                pos = adv_pos

        # 7. Colophon: इति संवत् … शुभम् terminates the document.
        colophon = _COLOPHON_RE.search(text, pos)
        middle_end = colophon.start() if colophon else len(text)

        # 6. Order: the trailing run of consecutively numbered line-start items
        # (the dispositive section) at the tail of the middle region.
        middle = text[pos:middle_end]
        numbered = [
            (m.start(), _devanagari_int(m.group(1)))
            for m in _NUMBERED_ITEM_RE.finditer(middle)
        ]
        numbered = [(p, n) for p, n in numbered if n is not None]
        order_start: int | None = None
        if numbered:
            run_start = len(numbered) - 1
            while (
                run_start > 0
                and numbered[run_start - 1][1] == numbered[run_start][1] - 1
            ):
                run_start -= 1
            order_start = numbered[run_start][0]

        # 5. Opinion: न्या. opener through the start of the order block.
        pre_order = middle[:order_start] if order_start is not None else middle
        opinion = _OPINION_RE.search(pre_order)
        if opinion:
            if pre_order[: opinion.start()].strip():
                self._emit(chunks, pre_order[: opinion.start()], "body")
            self._emit(chunks, pre_order[opinion.start() :], "opinion")
        elif pre_order.strip():
            # Unmarked narration (§1.1: 2/6 samples have no न्या. attribution).
            self._emit(chunks, pre_order, "body")

        if order_start is not None and middle[order_start:].strip():
            self._emit(chunks, middle[order_start:], "order")

        if colophon:
            self._emit(chunks, text[colophon.start() :], "colophon")

        return chunks

    def _emit(self, chunks: list[NKPChunk], text: str, section_type: str) -> None:
        for piece in _secondary_split(text.strip()):
            chunks.append(
                NKPChunk(
                    chunk_index=len(chunks),
                    chunk_text=piece,
                    embed_text=piece,
                    section_type=section_type,
                    section_label=SECTION_LABELS[section_type],
                )
            )
