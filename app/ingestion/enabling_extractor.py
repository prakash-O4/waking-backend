"""
Enabling-power extractor for subordinate regulations (नियमावली/नियमहरू).

Parses the preamble of a regulation, identifies the parent ऐन clause that
authorises it, and writes a `work_relations` row. The extractor is deterministic
— no LLM — and always emits exactly one row per document (resolved, unresolved,
or an explicit no-match sentinel).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Standard: "ऐन, २०७४ को दफा ४४ ले दिएको अधिकार"
_ENABLING_STRICT_RE = re.compile(
    r"([^\n।]{5,100})\s+को\s+(दफा|धारा)\s+([^\s,।]{1,20})"
    r"\s+ले\s+दिएको\s+अधिकार"
)

# उपदफा variant: "ऐन को दफा ४४ को उपदफा (२) ले दिएको अधिकार"
_ENABLING_UPADAFA_RE = re.compile(
    r"([^\n।]{5,100})\s+को\s+(दफा|धारा)\s+([^\s,।]{1,20})"
    r"\s+को\s+उपदफा\s+\(([^\)]{1,10})\)\s+ले\s+दिएको\s+अधिकार"
)

_AMEND_TAG_RE = re.compile(r"</?amend[^>]*>", re.IGNORECASE)

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _strip_amend_markup(text: str) -> str:
    return _AMEND_TAG_RE.sub("", text)


def _normalize_title(text: str) -> str:
    text = re.sub(r",\s*", " ", text)
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def _normalize_section_num(s: str) -> str:
    return s.strip().translate(_DEVA_DIGITS)


def _parse_enabling_clause(
    preamble: str,
) -> tuple[str, str, str | None, str, str] | None:
    """
    Return (provision_type_raw, section_num_raw, subsection_num_raw, act_ref_raw,
    provision_kind) or None when no enabling clause is found.

    *provision_type_raw* is the verbatim Nepali word ('दफा' or 'धारा').
    *provision_kind* is the normalized DB value ('dafa' or 'dhara').

    The उपदफा variant is checked first because it is more specific.
    """
    match = _ENABLING_UPADAFA_RE.search(preamble)
    if match:
        act_ref_raw = match.group(1).strip()
        provision_type_raw = match.group(2)
        provision_kind = "dafa" if provision_type_raw == "दफा" else "dhara"
        section_num_raw = match.group(3).strip()
        subsection_num_raw = match.group(4).strip()
        return (
            provision_type_raw,
            section_num_raw,
            subsection_num_raw,
            act_ref_raw,
            provision_kind,
        )

    match = _ENABLING_STRICT_RE.search(preamble)
    if match:
        act_ref_raw = match.group(1).strip()
        provision_type_raw = match.group(2)
        provision_kind = "dafa" if provision_type_raw == "दफा" else "dhara"
        section_num_raw = match.group(3).strip()
        return provision_type_raw, section_num_raw, None, act_ref_raw, provision_kind

    return None


def _resolve_work(conn: Any, normalized_title: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM work WHERE title_ne = %s LIMIT 1",
            (normalized_title,),
        )
        row = cur.fetchone()
    return str(row[0]) if row else None


def _insert_relation(
    conn: Any,
    subordinate_work_id: str,
    enabling_work_id: str | None,
    provision_type: str | None,
    section_num: str | None,
    subsection_num: str | None,
    raw_clause_text: str,
    status: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO work_relations (
                subordinate_work_id, relation_type, enabling_work_id,
                enabling_provision_type, enabling_section_number,
                enabling_subsection_number, raw_clause_text, resolution_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                subordinate_work_id,
                "enabling_power",
                enabling_work_id,
                provision_type,
                section_num,
                subsection_num,
                raw_clause_text,
                status,
            ),
        )


def extract_enabling_clause(
    content: str,
    work_id: str,
    conn: Any,
    source_id: str = "",
) -> None:
    """
    Extract an enabling clause from the first 1500 chars of *content* and write
    it to `work_relations`. Always writes exactly one row (resolved, unresolved,
    or no_enabling_clause sentinel). Idempotent via ON CONFLICT DO NOTHING.
    """
    preamble = _strip_amend_markup(content[:1500])
    row = _parse_enabling_clause(preamble)

    if row is None:
        _insert_relation(
            conn,
            work_id,
            None,
            None,
            None,
            None,
            raw_clause_text="",
            status="no_enabling_clause",
        )
        return

    (
        provision_type_raw,
        section_num_raw,
        subsection_num_raw,
        act_ref_raw,
        provision_kind,
    ) = row
    section_num = _normalize_section_num(section_num_raw)
    subsection_num = (
        _normalize_section_num(subsection_num_raw) if subsection_num_raw else None
    )
    act_ref_norm = _normalize_title(act_ref_raw)

    enabling_work_id = _resolve_work(conn, act_ref_norm)
    status = "auto_extracted" if enabling_work_id else "parent_not_in_corpus"

    _insert_relation(
        conn,
        work_id,
        enabling_work_id,
        provision_kind,
        section_num,
        subsection_num,
        raw_clause_text=f"{act_ref_raw} को {provision_type_raw} {section_num_raw}",
        status=status,
    )
