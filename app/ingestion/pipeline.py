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
import math
import re
import time
import unicodedata
from datetime import date
from typing import Any

from psycopg2.extensions import connection as PgConnection

from app.authority.bs_ad_calendar import BeyondCalendarRange, lookup
from app.authority.parser import parse_law
from app.authority.writer import (
    upsert_component,
    upsert_expression,
    upsert_source,
    upsert_work,
)
from app.ingestion import metadata_enricher
from app.ingestion.enabling_extractor import extract_enabling_clause
from app.ingestion.laws_chunker import LawsChunker
from app.ingestion.nkp_chunker import NKPChunker
from app.ingestion.pgvector_indexer import PgvectorIndexer
from app.ingestion.pii_redactor import PIIRedactor, RedactionVerificationError
from app.ingestion.tariff_chunker import TariffChunker, is_tariff_dominant
from app.utils.loggers import logger

_DAFA_ANCHOR_RE = re.compile(r"\*\*[०-९]+\.")
_BS_DATE_RE = re.compile(r"([०-९]{4})[।./-]([०-९]{1,2})[।./-]([०-९]{1,2})")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_LANDMARK_BENCHES = {"पूर्ण इजलास", "संवैधानिक इजलास"}
_NIYAM_RE = re.compile(r"(?:^|\s)(नियमावली|नियमहरू|नियम)(?=\s|$|,|\.)")


_lf_client: Any = None


def _get_lf_client() -> Any | None:
    from app.config import get_settings

    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return None
    global _lf_client
    if _lf_client is None:
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "LANGFUSE_PUBLIC_KEY set but langfuse not installed — tracing disabled"
            )
            return None

        settings = get_settings()
        _lf_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    return _lf_client


def _begin_span(trace: Any, stage: str, input: dict) -> Any:
    if trace is None:
        return None
    try:
        return trace.span(name=f"stage.{stage}", input=input)
    except Exception:
        return None


def _end_span(span: Any, output: dict) -> None:
    if span is None:
        return
    try:
        span.end(output=output)
    except Exception:
        pass


def _fmt_latency(s: float) -> str:
    return f"{int(s * 1000)}ms" if s < 1.0 else f"{s:.1f}s"


def _end_trace(trace: Any, output: dict | None = None) -> None:
    if trace is None:
        return
    try:
        if output is not None:
            trace.update(output=output)
        trace.end()
    except Exception:
        pass


def _flush(lf: Any) -> None:
    if lf:
        lf.flush()


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
        self._tariff_chunker = TariffChunker()
        self._enable_llm = enable_llm
        # Outcome of the most recent ingest call, for CLI reporting:
        # 'ingested' | 'skipped' | 'rejected' | 'quarantined'.
        self.last_outcome = ""

    # ------------------------------------------------------------------ laws

    def ingest_law(self, record: dict[str, Any]) -> str | None:
        """Run stages 1–8 for a laws.jsonl record. Returns document_id or None."""
        record_start = time.monotonic()
        content = str(record.get("content") or "")
        source_type = str(record.get("document_type") or "act")
        if source_type not in ("act", "regulation"):
            source_type = "act"
        source_id = str(record.get("_id") or record.get("name") or "")
        content_hash = _content_hash(content)

        # Idempotency check before any Langfuse trace is created — skipped
        # records produce no trace (no noise in observability dashboard).
        t0 = time.monotonic()
        existing = self._find_existing(source_type, source_id)
        elapsed = time.monotonic() - t0
        if existing and existing[1] == content_hash:
            logger.info(f"{source_id}: unchanged content_hash, skipping")
            self.last_outcome = "skipped"
            print(
                f"  {'LOAD':<12} {_fmt_latency(elapsed)}  skipped (unchanged)",
                flush=True,
            )
            return None

        lf = _get_lf_client()
        trace = (
            lf.trace(
                name="ingestion.law",
                input={
                    "source_id": source_id,
                    "source_type": source_type,
                    "content_len": len(content),
                },
                metadata={"source_id": source_id, "source_type": source_type},
            )
            if lf
            else None
        )

        t0 = time.monotonic()
        span = _begin_span(
            trace,
            "LOAD",
            {
                "source_id": source_id,
                "source_type": source_type,
                "content_len": len(content),
            },
        )

        law = parse_law(record)
        work_id = upsert_work(self._conn, law)
        if existing:
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
        _end_span(span, {"outcome": "passed", "is_amendment": bool(existing)})
        print(
            f"  {'LOAD':<12} {_fmt_latency(elapsed)}  new document · {len(content):,} chars",
            flush=True,
        )

        t0 = time.monotonic()
        span = _begin_span(trace, "VALIDATE", {"content_len": len(content)})
        has_dafa = bool(content.strip() and _DAFA_ANCHOR_RE.search(content))
        elapsed = time.monotonic() - t0
        if not has_dafa:
            logger.warning(f"{source_id}: no दफा heading found, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _end_span(span, {"outcome": "rejected", "has_dafa_anchor": False})
            print(
                f"  {'VALIDATE':<12} {_fmt_latency(elapsed)}  दफा anchor missing",
                flush=True,
            )
            print("  ✗ rejected", flush=True)
            _end_trace(trace, {"outcome": "rejected", "chunk_count": 0})
            _flush(lf)
            return None
        _end_span(span, {"outcome": "passed", "has_dafa_anchor": True})
        print(
            f"  {'VALIDATE':<12} {_fmt_latency(elapsed)}  दफा anchor found", flush=True
        )

        t0 = time.monotonic()
        span = _begin_span(
            trace,
            "PERSIST_AUTHORITY",
            {"work_id": work_id, "component_count": len(law.components)},
        )
        try:
            upsert_source(self._conn, work_id, law, source_url=None)
            today = date.today()
            for component in law.components:
                upsert_component(self._conn, work_id, component)
                upsert_expression(self._conn, component, as_of=today)
        except Exception as exc:  # noqa: BLE001 - authority write failure rejects ingest
            logger.warning(f"{source_id}: authority persistence failed: {exc}")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            elapsed = time.monotonic() - t0
            _end_span(
                span,
                {
                    "outcome": "rejected",
                    "component_count": len(law.components),
                    "error": str(exc),
                },
            )
            print(
                f"  {'PERSIST_AUTH':<12} {_fmt_latency(elapsed)}  rejected",
                flush=True,
            )
            print("  ✗ rejected", flush=True)
            _end_trace(trace, {"outcome": "rejected", "chunk_count": 0})
            _flush(lf)
            return None
        elapsed = time.monotonic() - t0
        _end_span(
            span, {"outcome": "passed", "component_count": len(law.components)}
        )
        print(
            f"  {'PERSIST_AUTH':<12} {_fmt_latency(elapsed)}  {len(law.components)} components",
            flush=True,
        )

        t0 = time.monotonic()
        span = _begin_span(trace, "CHUNK", {"content_len": len(content)})
        is_tariff = is_tariff_dominant(content)
        if is_tariff:
            chunks = self._tariff_chunker.chunk_text(
                content, act_name=record.get("name", "")
            )
        else:
            chunks = self._laws_chunker.chunk_text(content)
        elapsed = time.monotonic() - t0
        if not chunks:
            logger.warning(f"{source_id}: chunker produced no chunks, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _end_span(span, {"outcome": "rejected", "chunk_count": 0})
            print(f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → 0 chunks", flush=True)
            print("  ✗ rejected", flush=True)
            _end_trace(trace, {"outcome": "rejected", "chunk_count": 0})
            _flush(lf)
            return None
        _end_span(
            span,
            {
                "outcome": "passed",
                "chunk_count": len(chunks),
                "chunker": "tariff" if is_tariff else "laws",
            },
        )
        print(
            f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → {len(chunks)} chunks",
            flush=True,
        )

        doc_name = str(record.get("name") or "")
        if _NIYAM_RE.search(doc_name):
            try:
                extract_enabling_clause(
                    content=content,
                    work_id=work_id,
                    conn=self._conn,
                    source_id=source_id,
                )
            except Exception as exc:  # noqa: BLE001 — derivative metadata, never block ingest
                logger.warning(
                    f"{source_id}: enabling-power extraction failed: {exc}"
                )

        batch_count = math.ceil(len(chunks) / metadata_enricher.CHUNK_BATCH_SIZE)
        span = _begin_span(
            trace,
            "EXTRACT_METADATA",
            {
                "chunk_count": len(chunks),
                "batch_count": batch_count,
                "llm_calls": 0
                if is_tariff
                else (1 + batch_count if self._enable_llm else 0),
            },
        )
        t0 = time.monotonic()
        metadata_outcome = "passed"
        summary: str | None = None
        llm_calls = in_tok = out_tok = 0
        if is_tariff:
            summary = f"{record.get('name', '')} — भन्सार महसुल दर तालिका"
        elif self._enable_llm:
            try:
                metadata, llm_calls, in_tok, out_tok = (
                    metadata_enricher.enrich_law_chunks(record, chunks, lf_parent=span)
                )
                for meta in metadata:
                    index = meta["chunk_index"]
                    chunks[index].keywords = meta.get("keywords")
                    chunks[index].relevant_questions = meta.get("relevant_questions")
                    summary = meta.get("summary") or summary
            except Exception as exc:  # noqa: BLE001 — NULL columns, never crash
                metadata_outcome = "failed"
                logger.warning(f"{source_id}: metadata extraction failed: {exc}")
        elapsed = time.monotonic() - t0
        _end_span(
            span,
            {
                "outcome": metadata_outcome,
                "llm_calls": llm_calls,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "summary_extracted": bool(summary),
                "keywords_extracted": sum(1 for c in chunks if c.keywords),
            },
        )
        print(
            f"  {'EXTRACT_META':<12} {_fmt_latency(elapsed)}  {llm_calls} LLM calls · {in_tok:,} in + {out_tok:,} out tok",
            flush=True,
        )

        for chunk in chunks:
            chunk.effective_date_ad = self._commence_date(law.uri, chunk.section_number)

        embed_span = _begin_span(
            trace, "EMBED_AND_UPSERT", {"chunk_count": len(chunks)}
        )
        t0 = time.monotonic()
        embeddings, embed_tok = self._embed(
            [c.embed_text for c in chunks], lf_parent=embed_span
        )
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
        elapsed = time.monotonic() - t0
        _end_span(
            embed_span,
            {
                "outcome": "passed",
                "document_id": document_id,
                "total_embedding_tokens": embed_tok,
            },
        )
        print(
            f"  {'EMBED+UPSERT':<12} {_fmt_latency(elapsed)}  {len(chunks)} chunks · {embed_tok:,} embed tok",
            flush=True,
        )

        span = _begin_span(trace, "DUAL_APPROVAL_PAUSE", {"document_id": document_id})
        t0 = time.monotonic()
        logger.info(f"document {document_id} awaiting dual approval.")
        self.last_outcome = "ingested"
        elapsed = time.monotonic() - t0
        _end_span(span, {"outcome": "ingested", "status": "pending"})
        print(f"  {'DUAL_APPROVAL':<12} {_fmt_latency(elapsed)}  pending", flush=True)
        _end_trace(
            trace,
            {
                "outcome": "ingested",
                "chunk_count": len(chunks),
                "total_llm_calls": llm_calls,
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
                "total_embedding_tokens": embed_tok,
            },
        )
        print(
            f"  {'total':<12} {_fmt_latency(time.monotonic() - record_start)}",
            flush=True,
        )
        _flush(lf)
        return document_id

    # -------------------------------------------------------------- nkp cases

    def ingest_nkp_case(self, record: dict[str, Any]) -> str | None:
        """Run stages 1–8 for an nkp_cases.jsonl record."""
        record_start = time.monotonic()
        full_text = str(record.get("full_text") or "")
        source_id = str(record.get("case_id") or "")
        content_hash = _content_hash(full_text)

        # Idempotency check before any Langfuse trace is created — skipped
        # records produce no trace (no noise in observability dashboard).
        t0 = time.monotonic()
        existing = self._find_existing("nkp_case", source_id)
        elapsed = time.monotonic() - t0
        if existing and existing[1] == content_hash:
            logger.info(f"{source_id}: unchanged content_hash, skipping")
            self.last_outcome = "skipped"
            print(
                f"  {'LOAD':<12} {_fmt_latency(elapsed)}  skipped (unchanged)",
                flush=True,
            )
            return None

        lf = _get_lf_client()
        trace = (
            lf.trace(
                name="ingestion.nkp_case",
                input={
                    "source_id": source_id,
                    "source_type": "nkp_case",
                    "content_len": len(full_text),
                },
                metadata={"source_id": source_id, "source_type": "nkp_case"},
            )
            if lf
            else None
        )

        t0 = time.monotonic()
        span = _begin_span(
            trace,
            "LOAD",
            {
                "source_id": source_id,
                "source_type": "nkp_case",
                "content_len": len(full_text),
            },
        )
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
            document_id = self._insert_document("nkp_case", source_id, content_hash, "")
        _end_span(span, {"outcome": "passed", "is_amendment": bool(existing)})
        print(
            f"  {'LOAD':<12} {_fmt_latency(elapsed)}  new document · {len(full_text):,} chars",
            flush=True,
        )

        t0 = time.monotonic()
        span = _begin_span(trace, "VALIDATE", {"content_len": len(full_text)})
        is_valid = bool(full_text.strip())
        elapsed = time.monotonic() - t0
        if not is_valid:
            logger.warning(f"{source_id}: empty full_text, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _end_span(span, {"outcome": "rejected", "has_dafa_anchor": False})
            print(
                f"  {'VALIDATE':<12} {_fmt_latency(elapsed)}  empty full_text",
                flush=True,
            )
            print("  ✗ rejected", flush=True)
            _end_trace(trace, {"outcome": "rejected", "chunk_count": 0})
            _flush(lf)
            return None
        _end_span(span, {"outcome": "passed", "has_dafa_anchor": False})
        print(f"  {'VALIDATE':<12} {_fmt_latency(elapsed)}  non-empty text", flush=True)

        t0 = time.monotonic()
        span = _begin_span(trace, "REDACT_PII", {"content_len": len(full_text)})
        appellant = str(record.get("appellant") or "")
        respondent = str(record.get("respondent") or "")
        try:
            redacted_text, warnings = self._redactor.redact(
                full_text, appellant, respondent, document_id=source_id
            )
            for warning in warnings:
                logger.warning(f"{source_id}: redaction warning: {warning}")
        except RedactionVerificationError as exc:
            logger.error(f"{source_id}: redaction verification failed: {exc}")
            self._execute(
                "UPDATE documents SET redaction_failed = TRUE WHERE id = %s",
                (document_id,),
            )
            self._conn.commit()
            self.last_outcome = "quarantined"
            elapsed = time.monotonic() - t0
            _end_span(span, {"outcome": "quarantined", "warning_count": 0})
            print(
                f"  {'REDACT_PII':<12} {_fmt_latency(elapsed)}  quarantined", flush=True
            )
            _end_trace(trace, {"outcome": "quarantined", "chunk_count": 0})
            _flush(lf)
            return None
        elapsed = time.monotonic() - t0
        _end_span(
            span,
            {
                "outcome": "passed",
                "warning_count": len(warnings),
                "content_len": len(redacted_text),
            },
        )
        print(
            f"  {'REDACT_PII':<12} {_fmt_latency(elapsed)}  {len(warnings)} warnings",
            flush=True,
        )

        t0 = time.monotonic()
        span = _begin_span(trace, "CHUNK", {"content_len": len(redacted_text)})
        chunks = self._nkp_chunker.chunk_text(redacted_text)
        elapsed = time.monotonic() - t0
        if not chunks:
            logger.warning(f"{source_id}: chunker produced no chunks, rejecting")
            self._set_status(document_id, "rejected")
            self._conn.commit()
            self.last_outcome = "rejected"
            _end_span(span, {"outcome": "rejected", "chunk_count": 0})
            print(f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → 0 chunks", flush=True)
            print("  ✗ rejected", flush=True)
            _end_trace(trace, {"outcome": "rejected", "chunk_count": 0})
            _flush(lf)
            return None
        _end_span(span, {"outcome": "passed", "chunk_count": len(chunks)})
        print(
            f"  {'CHUNK':<12} {_fmt_latency(elapsed)}  → {len(chunks)} chunks",
            flush=True,
        )

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

        batch_count = math.ceil(len(chunks) / metadata_enricher.CHUNK_BATCH_SIZE)
        span = _begin_span(
            trace,
            "EXTRACT_METADATA",
            {
                "chunk_count": len(chunks),
                "batch_count": batch_count,
                "llm_calls": 1 + batch_count if self._enable_llm else 0,
            },
        )
        t0 = time.monotonic()
        metadata_outcome = "passed"
        cited_statutes: list[str] | None = None
        headnotes: str | None = None
        summary: str | None = None
        llm_calls = in_tok = out_tok = 0
        if self._enable_llm:
            try:
                metadata, llm_calls, in_tok, out_tok = (
                    metadata_enricher.enrich_nkp_chunks(record, chunks, lf_parent=span)
                )
                for meta in metadata:
                    index = meta["chunk_index"]
                    chunks[index].keywords = meta.get("keywords")
                    chunks[index].relevant_questions = meta.get("relevant_questions")
                    cited_statutes = meta.get("cited_statutes") or cited_statutes
                    headnotes = meta.get("headnotes") or headnotes
                    summary = meta.get("summary") or summary
            except Exception as exc:  # noqa: BLE001 — NULL columns, never crash
                metadata_outcome = "failed"
                logger.warning(f"{source_id}: metadata extraction failed: {exc}")
            cited_statutes = self._validate_cited_statutes(cited_statutes)
        elapsed = time.monotonic() - t0
        _end_span(
            span,
            {
                "outcome": metadata_outcome,
                "llm_calls": llm_calls,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "summary_extracted": bool(summary),
                "keywords_extracted": sum(1 for c in chunks if c.keywords),
            },
        )
        print(
            f"  {'EXTRACT_META':<12} {_fmt_latency(elapsed)}  {llm_calls} LLM calls · {in_tok:,} in + {out_tok:,} out tok",
            flush=True,
        )

        embed_span = _begin_span(
            trace, "EMBED_AND_UPSERT", {"chunk_count": len(chunks)}
        )
        t0 = time.monotonic()
        embeddings, embed_tok = self._embed(
            [c.embed_text for c in chunks], lf_parent=embed_span
        )
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
        elapsed = time.monotonic() - t0
        _end_span(
            embed_span,
            {
                "outcome": "passed",
                "document_id": document_id,
                "total_embedding_tokens": embed_tok,
            },
        )
        print(
            f"  {'EMBED+UPSERT':<12} {_fmt_latency(elapsed)}  {len(chunks)} chunks · {embed_tok:,} embed tok",
            flush=True,
        )

        span = _begin_span(trace, "DUAL_APPROVAL_PAUSE", {"document_id": document_id})
        t0 = time.monotonic()
        logger.info(f"document {document_id} awaiting dual approval.")
        self.last_outcome = "ingested"
        elapsed = time.monotonic() - t0
        _end_span(span, {"outcome": "ingested", "status": "pending"})
        print(f"  {'DUAL_APPROVAL':<12} {_fmt_latency(elapsed)}  pending", flush=True)
        _end_trace(
            trace,
            {
                "outcome": "ingested",
                "chunk_count": len(chunks),
                "total_llm_calls": llm_calls,
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
                "total_embedding_tokens": embed_tok,
            },
        )
        print(
            f"  {'total':<12} {_fmt_latency(time.monotonic() - record_start)}",
            flush=True,
        )
        _flush(lf)
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

    def _embed(
        self, texts: list[str], lf_parent: Any = None
    ) -> tuple[list[list[float]], int]:
        return self._indexer.embed_chunks(texts, lf_parent=lf_parent)
