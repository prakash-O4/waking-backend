"""
pgvector indexer — replaces pinecone_indexer.py.

Upserts documents + chunks (with bge-m3 embeddings) into the PostgreSQL
search-derivative tables from migrations/005_ingestion_pipeline.sql. The
bitemporal store remains the only authority; these tables are derivatives
(system-design §2 invariant 1).
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from typing import Any

from openai import AzureOpenAI
from psycopg2.extensions import connection as PgConnection

from app.ingestion.laws_chunker import LawChunk
from app.ingestion.nkp_chunker import NKPChunk
from app.utils.loggers import logger

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1024 
DEFAULT_BATCH_SIZE = 512  # Azure API has no GPU memory limit — large batches reduce round-trips

_CHUNK_INSERT_SQL = """
INSERT INTO chunks (
    id, document_id, source_type, chunk_index, chunk_text, span_sha256,
    embedding, chunk_type,
    case_id, case_type, court, bench_type, decision_date_ad, year_bs,
    section_type, is_landmark, parties_redacted, cited_statutes, headnotes,
    work_id, act_name, english_name, document_type, section_number,
    section_title, chapter_number, level, parent_section, effective_date_ad,
    co_retrieve_parent_id, keywords, relevant_questions
) VALUES (
    %(id)s, %(document_id)s, %(source_type)s, %(chunk_index)s, %(chunk_text)s,
    %(span_sha256)s, %(embedding)s, %(chunk_type)s,
    %(case_id)s, %(case_type)s, %(court)s, %(bench_type)s, %(decision_date_ad)s,
    %(year_bs)s, %(section_type)s, %(is_landmark)s, %(parties_redacted)s,
    %(cited_statutes)s, %(headnotes)s,
    %(work_id)s, %(act_name)s, %(english_name)s, %(document_type)s,
    %(section_number)s, %(section_title)s, %(chapter_number)s, %(level)s,
    %(parent_section)s, %(effective_date_ad)s, %(co_retrieve_parent_id)s,
    %(keywords)s, %(relevant_questions)s
)
"""


class PgvectorIndexer:
    def __init__(self, conn: PgConnection):
        self._conn = conn
        self._embed_client: AzureOpenAI | None = None
        try:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
        except Exception:  # noqa: BLE001 — ImportError when absent, ProgrammingError on mocked connections
            # pgvector is only needed for real DB writes; unit tests run with
            # mocked connections and without the package installed.
            logger.warning("pgvector vector adapter not registered (mocked connection or package missing)")

    def _get_embed_client(self) -> AzureOpenAI:
        if self._embed_client is None:
            from app.config import azure_base_url, get_settings

            s = get_settings()
            self._embed_client = AzureOpenAI(
                api_key=s.AZURE_OPENAI_KEY,
                api_version=s.AZURE_OPENAI_API_VERSION,
                azure_endpoint=azure_base_url(),
            )
        return self._embed_client

    def embed_chunks(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed in batches via Azure OpenAI text-embedding-3-large (dimensions=1024)."""
        if not texts:
            return []
        from app.config import get_settings

        client = self._get_embed_client()
        deployment = get_settings().AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        dims = get_settings().AZURE_OPENAI_EMBEDDING_DIMENSIONS
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(
                model=deployment,
                input=batch,
                dimensions=dims,
            )
            embeddings.extend([d.embedding for d in response.data])
        return embeddings

    def upsert_document(
        self,
        document: dict[str, Any],
        chunks: list[LawChunk] | list[NKPChunk],
        embeddings: list[list[float]],
    ) -> str:
        """
        Insert the documents row + all chunk rows in one transaction (the
        caller's connection; commit happens on conn.commit()). Returns the
        document_id (UUID as str). Chunks are inserted in chunk_index order so
        co_retrieve_parent_id always references an already-inserted row.
        On UNIQUE(document_id, chunk_index) conflict: existing chunks for this
        document are deleted first, then re-inserted.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    source_type, source_id, content_hash, raw_content,
                    ocr_confidence, summary
                ) VALUES (%(source_type)s, %(source_id)s, %(content_hash)s,
                          %(raw_content)s, %(ocr_confidence)s, %(summary)s)
                ON CONFLICT (source_type, source_id) DO UPDATE
                SET content_hash = EXCLUDED.content_hash,
                    raw_content = EXCLUDED.raw_content,
                    summary = EXCLUDED.summary
                RETURNING id
                """,
                {
                    "source_type": document["source_type"],
                    "source_id": document["source_id"],
                    "content_hash": document["content_hash"],
                    "raw_content": document["raw_content"],
                    "ocr_confidence": document.get("ocr_confidence"),
                    "summary": document.get("summary"),
                },
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("documents upsert returned no id")
            document_id = str(row[0])

            # Idempotent re-ingest: replace this document's chunks wholesale.
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

            # Pre-generate chunk UUIDs so co_retrieve_parent_index can resolve
            # to the parent's UUID before insert.
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            for chunk, embedding, chunk_uuid in zip(chunks, embeddings, chunk_ids):
                cur.execute(
                    _CHUNK_INSERT_SQL,
                    self._chunk_row(document, document_id, chunk, embedding, chunk_ids),
                )
        return document_id

    def insert_pii_vault(
        self, document_id: str, appellant: str, respondent: str, full_text_raw: str
    ) -> None:
        """Write to pii_vault. Called only for nkp_case source_type (PS-14)."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pii_vault (document_id, appellant_raw, respondent_raw,
                                       full_text_raw)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET appellant_raw = EXCLUDED.appellant_raw,
                    respondent_raw = EXCLUDED.respondent_raw,
                    full_text_raw = EXCLUDED.full_text_raw
                """,
                (document_id, appellant, respondent, full_text_raw),
            )

    def _chunk_row(
        self,
        document: dict[str, Any],
        document_id: str,
        chunk: LawChunk | NKPChunk,
        embedding: list[float],
        chunk_ids: list[str],
    ) -> dict[str, Any]:
        span_sha256 = hashlib.sha256(
            unicodedata.normalize("NFC", chunk.chunk_text).encode("utf-8")
        ).hexdigest()
        row: dict[str, Any] = {
            "id": chunk_ids[chunk.chunk_index],
            "document_id": document_id,
            "source_type": document["source_type"],
            "chunk_index": chunk.chunk_index,
            "chunk_text": chunk.chunk_text,
            "span_sha256": span_sha256,
            "embedding": embedding,
            "keywords": chunk.keywords,
            "relevant_questions": chunk.relevant_questions,
            # NKP columns (NULL for law rows)
            "case_id": None,
            "case_type": None,
            "court": None,
            "bench_type": None,
            "decision_date_ad": None,
            "year_bs": None,
            "section_type": None,
            "is_landmark": None,
            "parties_redacted": None,
            "cited_statutes": None,
            "headnotes": None,
            # law columns (NULL for nkp_case rows)
            "work_id": None,
            "act_name": None,
            "english_name": None,
            "document_type": None,
            "section_number": None,
            "section_title": None,
            "chapter_number": None,
            "level": None,
            "parent_section": None,
            "effective_date_ad": None,
            "co_retrieve_parent_id": None,
            "chunk_type": "",
        }
        if isinstance(chunk, NKPChunk):
            row.update(
                {
                    "chunk_type": chunk.section_label,
                    "case_id": document.get("case_id"),
                    "case_type": document.get("case_type"),
                    "court": document.get("court"),
                    "bench_type": document.get("bench_type"),
                    "decision_date_ad": document.get("decision_date_ad"),
                    "year_bs": document.get("year_bs"),
                    "section_type": chunk.section_type,
                    "is_landmark": document.get("is_landmark"),
                    "parties_redacted": document.get("parties_redacted"),
                    "cited_statutes": document.get("cited_statutes"),
                    "headnotes": document.get("headnotes"),
                }
            )
        else:
            row.update(
                {
                    "chunk_type": (
                        f"दफा {chunk.section_number}"
                        if chunk.section_number
                        else chunk.level
                    ),
                    "work_id": document.get("work_id"),
                    "act_name": document.get("act_name"),
                    "english_name": document.get("english_name"),
                    "document_type": document.get("document_type"),
                    "section_number": chunk.section_number,
                    "section_title": chunk.section_title,
                    "chapter_number": chunk.chapter_number,
                    "level": chunk.level,
                    "parent_section": chunk.parent_section,
                    "effective_date_ad": chunk.effective_date_ad,
                    "co_retrieve_parent_id": (
                        chunk_ids[chunk.co_retrieve_parent_index]
                        if chunk.co_retrieve_parent_index is not None
                        else None
                    ),
                }
            )
        return row
