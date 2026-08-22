"""
8-stage ingestion pipeline orchestrator (design §6).

LOAD → VALIDATE → REDACT_PII (nkp only) → CHUNK → EXTRACT_METADATA → EMBED →
UPSERT → DUAL APPROVAL PAUSE. The pipeline never sets ingestion_status to
'approved' — that flip is human-only via a separate admin path (PS-2). A loud
refusal beats a quiet wrong answer: validation failures reject, redaction
failures quarantine, LLM failures leave NULL columns.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from typing import Any

from psycopg2.extensions import connection as PgConnection

from app.authority.bs_ad_calendar import BeyondCalendarRange, lookup
from app.authority.parser import parse_law
from app.authority.writer import upsert_work
from app.ingestion import metadata_enricher
from app.ingestion.laws_chunker import LawsChunker
from app.ingestion.nkp_chunker import NKPChunker
from app.ingestion.pgvector_indexer import PgvectorIndexer
from app.ingestion.pii_redactor import PIIRedactor, RedactionVerificationError
from app.utils.loggers import logger

_DAFA_ANCHOR_RE = re.compile(r"\*\*[०-९]+\.")
_BS_DATE_RE = re.compile(r"([०-९]{4})[।./-]([०-९]{1,2})[।./-]([०-९]{1,2})")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_LANDMARK_BENCHES = {"पूर्ण इजलास", "संवैधानिक इजलास"}


def _emit_ingestion_span(
    source_id: str, source_type: str, stage: str, outcome: str
) -> None:
    from app.config import Settings

    settings = Settings()
    if not settings.LANGFUSE_PUBLIC_KEY:
        return
    from langfuse import Langfuse  # type: ignore[import-not-found]

    metadata = {
        "source_id": source_id,
        "source_type": source_type,
        "stage": stage,
        "outcome": outcome,
    }
    client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    trace = client.trace(name="ingestion.document", metadata=metadata)
    span = trace.span(name=f"ingestion.{stage}", metadata=metadata)
    span.end()
    client.flush()


def _content_hash(content: str) -> str:
    return hashlib.sha256(
        unicodedata.normalize("NFC", content).encode("utf-8")
    ).hexdigest()


def _devanagari_to_int(s: str) -> int:
    return int("".join(str(ord(c) - ord("०")) for c in s))


def parse_decision_date(decision_date_bs: str) -> tuple[date | None, bool]:
    """
    Parses '२०८१/०१/०३' (Devanagari digits, BS) to AD date via the
    bs_ad_calendar authority module — never a date library (PS-5).
    Returns (ad_date, is_boundary_window).
    Returns (None, False) if unparseable or out of calendar range (log warning).
    """
    match = _BS_DATE_RE.search(decision_date_bs or "")
    if not match:
        logger.warning(f"unparseable decision_date: {decision_date_bs!r}")
        return None, False
    year, month, day = (_devanagari_to_int(group) for group in match.groups())
    try:
        ad_date, is_boundary = lookup(year, month, day)
    except BeyondCalendarRange as exc:
        logger.warning(f"decision_date beyond calendar range: {exc}")
        return None, False
    return ad_date, is_boundary


class IngestionPipeline:
    def __init__(self, conn: PgConnection, *, enable_llm: bool = True):
        self._conn = conn
        self._indexer = PgvectorIndexer(conn)
        self._redactor = PIIRedactor(enable_llm=enable_llm)
        self._laws_chunker = LawsChunker()
        self._nkp_chunker = NKPChunker()
        self._enable_llm = enable_llm
        # Outcome of the most recent ingest call, for CLI reporting:
        # 'ingested' | 'skipped' | 'rejected' | 'quarantined'.
        self.last_outcome = ""

    # ------------------------------------------------------------------ laws

    def ingest_law(self, record: dict[str, Any]) -> str | None:
        """Run stages 1–8 for a laws.jsonl record. Returns document_id or None."""
        content = str(record.get("content") or "")
        source_type = str(record.get("document_type") or "act")
        if source_type not in ("act", "regulation"):
            source_type = "act"
        source_id = str(record.get("_id") or record.get("name") or "")
        content_hash = _content_hash(content)

        # Stage 1 — LOAD (idempotency check before any paid work).
        existing = self._find_existing(source_type, source_id)
        if existing and existing[1] == content_hash:
            logger.info(f"{source_id}: unchanged content_hash, skipping")
            self.last_outcome = "skipped"
            _emit_ingestion_span(source_id, source_type, "LOAD", "skipped")
            return None

        # The work row is upserted during document load (design §3.2).
        law = parse_law(record)
        work_id = upsert_work(self._conn, law)

        if existing:
            # Amended source. UNIQUE(source_type, source_id) forbids a second
            # row, so the existing row is reset to pending with a fresh
            # valid_time range (the old range is closed by replacement) and
            # approvers are cleared — never an in-place approval carry-over.
            document_id = existing[0]
            self._execute(
                """
                UPDATE documents
                SET content_hash = %s, raw_content = %s,
                    ingestion_status = 'pending', redaction_failed = FALSE,
                    approved_by = NULL, second_approved_by = NULL,
                    valid_time = tstzrange(now(), NULL), ingested_at = now()
                WHERE id = %s
                """,
                (content_hash, content, document_id),
            )
        else:
            document_id = self._insert_document(
                source_type, source_id, content_hash, content
            )

        # Stage 2 — VALIDATE.
        if not content.strip() or not _DAFA_ANCHOR_RE.search(content):
            logger.warning(f"{source_id}: no दफा heading found, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _emit_ingestion_span(source_id, source_type, "VALIDATE", "rejected")
            return None

        # Stage 4 — CHUNK.
        chunks = self._laws_chunker.chunk_text(content)
        if not chunks:
            logger.warning(f"{source_id}: chunker produced no chunks, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _emit_ingestion_span(source_id, source_type, "CHUNK", "rejected")
            return None

        # Stage 5 — EXTRACT_METADATA (haiku; failures leave NULL columns).
        summary: str | None = None
        if self._enable_llm:
            try:
                for meta in metadata_enricher.enrich_law_chunks(record, chunks):
                    index = meta["chunk_index"]
                    chunks[index].keywords = meta.get("keywords")
                    chunks[index].relevant_questions = meta.get("relevant_questions")
                    summary = meta.get("summary") or summary
            except Exception as exc:  # noqa: BLE001 — NULL columns, never crash
                logger.warning(f"{source_id}: metadata extraction failed: {exc}")

        # effective_date_ad: denormalized cache of the approved commence effect,
        # NULL-pending when unverified (PS-2 — never parsed from text).
        for chunk in chunks:
            chunk.effective_date_ad = self._commence_date(law.uri, chunk.section_number)

        # Stages 6–7 — EMBED + UPSERT.
        embeddings = self._embed([c.embed_text for c in chunks])
        document = {
            "source_type": source_type,
            "source_id": source_id,
            "content_hash": content_hash,
            "raw_content": content,
            "ocr_confidence": None,
            "work_id": work_id,
            "act_name": record.get("name"),
            "english_name": record.get("english_name"),
            "document_type": record.get("document_type"),
            "summary": summary,
        }
        document_id = self._indexer.upsert_document(document, chunks, embeddings)
        self._conn.commit()

        # Stage 8 — DUAL APPROVAL PAUSE (PS-2): nothing is searchable yet.
        logger.info(f"document {document_id} awaiting dual approval.")
        self.last_outcome = "ingested"
        _emit_ingestion_span(source_id, source_type, "DUAL_APPROVAL_PAUSE", "ingested")
        return document_id

    # -------------------------------------------------------------- nkp cases

    def ingest_nkp_case(self, record: dict[str, Any]) -> str | None:
        """Run stages 1–8 for an nkp_cases.jsonl record."""
        full_text = str(record.get("full_text") or "")
        source_id = str(record.get("case_id") or "")
        content_hash = _content_hash(full_text)

        # Stage 1 — LOAD.
        existing = self._find_existing("nkp_case", source_id)
        if existing and existing[1] == content_hash:
            logger.info(f"{source_id}: unchanged content_hash, skipping")
            self.last_outcome = "skipped"
            _emit_ingestion_span(source_id, "nkp_case", "LOAD", "skipped")
            return None
        if existing:
            document_id = existing[0]
            self._execute(
                """
                UPDATE documents
                SET content_hash = %s, ingestion_status = 'pending',
                    redaction_failed = FALSE,
                    approved_by = NULL, second_approved_by = NULL,
                    valid_time = tstzrange(now(), NULL), ingested_at = now()
                WHERE id = %s
                """,
                (content_hash, document_id),
            )
        else:
            # raw_content is filled with the REDACTED text at upsert time;
            # unredacted NKP content only ever lands in pii_vault (PS-14).
            document_id = self._insert_document("nkp_case", source_id, content_hash, "")

        # Stage 2 — VALIDATE.
        if not full_text.strip():
            logger.warning(f"{source_id}: empty full_text, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _emit_ingestion_span(source_id, "nkp_case", "VALIDATE", "rejected")
            return None

        # Stage 3 — REDACT_PII.
        appellant = str(record.get("appellant") or "")
        respondent = str(record.get("respondent") or "")
        try:
            redacted_text, warnings = self._redactor.redact(
                full_text, appellant, respondent, document_id=source_id
            )
            for warning in warnings:
                logger.warning(f"{source_id}: redaction warning: {warning}")
        except RedactionVerificationError as exc:
            # Quarantine: stays pending + flagged, never silently passed.
            logger.error(f"{source_id}: redaction verification failed: {exc}")
            self._execute(
                "UPDATE documents SET redaction_failed = TRUE WHERE id = %s",
                (document_id,),
            )
            self._conn.commit()
            self.last_outcome = "quarantined"
            _emit_ingestion_span(source_id, "nkp_case", "REDACT_PII", "quarantined")
            return None

        # Stage 4 — CHUNK (on redacted text).
        chunks = self._nkp_chunker.chunk_text(redacted_text)
        if not chunks:
            logger.warning(f"{source_id}: chunker produced no chunks, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _emit_ingestion_span(source_id, "nkp_case", "CHUNK", "rejected")
            return None

        # Deterministic case metadata (design §3.1).
        decision_date_ad, is_boundary = parse_decision_date(
            str(record.get("decision_date") or "")
        )
        if is_boundary:
            logger.warning(f"{source_id}: boundary-window decision date → human review")
        year_bs = None
        if record.get("decision_date"):
            match = _BS_DATE_RE.search(str(record["decision_date"]))
            if match:
                year_bs = _devanagari_to_int(match.group(1))
        case_type = str(record.get("case_type") or "").strip().lstrip("- ").strip()
        bench_type = str(record.get("bench") or "").strip()
        headnote_count = sum(1 for c in chunks if c.section_type == "headnote")
        is_landmark = bench_type in _LANDMARK_BENCHES or headnote_count >= 2

        # Stage 5 — EXTRACT_METADATA.
        cited_statutes: list[str] | None = None
        headnotes: str | None = None
        summary: str | None = None
        if self._enable_llm:
            try:
                for meta in metadata_enricher.enrich_nkp_chunks(record, chunks):
                    index = meta["chunk_index"]
                    chunks[index].keywords = meta.get("keywords")
                    chunks[index].relevant_questions = meta.get("relevant_questions")
                    cited_statutes = meta.get("cited_statutes") or cited_statutes
                    headnotes = meta.get("headnotes") or headnotes
                    summary = meta.get("summary") or summary
            except Exception as exc:  # noqa: BLE001 — NULL columns, never crash
                logger.warning(f"{source_id}: metadata extraction failed: {exc}")
            cited_statutes = self._validate_cited_statutes(cited_statutes)

        # Stages 6–7 — EMBED + UPSERT (+ pii_vault for the raw text, PS-14).
        embeddings = self._embed([c.embed_text for c in chunks])
        document = {
            "source_type": "nkp_case",
            "source_id": source_id,
            "content_hash": content_hash,
            "raw_content": redacted_text,
            "ocr_confidence": None,
            "case_id": source_id,
            "case_type": case_type or None,
            "court": record.get("court"),
            "bench_type": bench_type or None,
            "decision_date_ad": decision_date_ad,
            "year_bs": year_bs,
            "is_landmark": is_landmark,
            "parties_redacted": "[[वादी]] / [[प्रतिवादी]]",
            "cited_statutes": cited_statutes,
            "headnotes": headnotes,
            "summary": summary,
        }
        document_id = self._indexer.upsert_document(document, chunks, embeddings)
        self._indexer.insert_pii_vault(document_id, appellant, respondent, full_text)
        self._conn.commit()

        # Stage 8 — DUAL APPROVAL PAUSE (PS-2).
        logger.info(f"document {document_id} awaiting dual approval.")
        self.last_outcome = "ingested"
        _emit_ingestion_span(source_id, "nkp_case", "DUAL_APPROVAL_PAUSE", "ingested")
        return document_id

    # ---------------------------------------------------------------- helpers

    def _find_existing(
        self, source_type: str, source_id: str
    ) -> tuple[str, str] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash FROM documents "
                "WHERE source_type = %s AND source_id = %s",
                (source_type, source_id),
            )
            row = cur.fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def _insert_document(
        self, source_type: str, source_id: str, content_hash: str, raw_content: str
    ) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source_type, source_id, content_hash,
                                       raw_content)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (source_type, source_id, content_hash, raw_content),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError("documents insert returned no id")
        return str(row[0])

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _set_status(self, document_id: str, status: str) -> None:
        self._execute(
            "UPDATE documents SET ingestion_status = %s WHERE id = %s",
            (status, document_id),
        )

    def _commence_date(self, work_uri: str, section_number: str | None) -> date | None:
        """Approved commence effect for this दफा's component; None when
        unverified (PS-2) or on any lookup failure (cache, never authority)."""
        if not section_number:
            return None
        component_uri = (
            f"{work_uri}/dafa/{section_number.translate(_DEVANAGARI_DIGITS)}"
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MIN(effective_date) FROM lifecycle_effect
                    WHERE component_uri = %s AND effect_type = 'commence'
                      AND approval_status = 'approved'
                    """,
                    (component_uri,),
                )
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001 — cache miss must not crash
            logger.warning(f"commence lookup failed for {component_uri}: {exc}")
            return None
        return row[0] if row and row[0] else None

    def _validate_cited_statutes(self, cited: list[str] | None) -> list[str] | None:
        """LLM-proposed citations are dropped unless they match work.title_ne
        (design §3.3 — proposals, never guessed)."""
        if not cited:
            return None
        validated: list[str] = []
        try:
            with self._conn.cursor() as cur:
                for title in cited:
                    cur.execute(
                        "SELECT 1 FROM work WHERE title_ne = %s LIMIT 1", (title,)
                    )
                    if cur.fetchone():
                        validated.append(title)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"cited_statutes validation failed, dropping: {exc}")
            return None
        return validated or None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._indexer.embed_chunks(texts)
