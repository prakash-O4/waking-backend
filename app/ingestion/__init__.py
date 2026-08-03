"""
Ingestion pipeline for Nepali legal documents (PE-A: PostgreSQL + pgvector).
"""

from .document_processor import DocumentProcessor
from .laws_chunker import LawsChunker
from .nkp_chunker import NKPChunker
from .pgvector_indexer import PgvectorIndexer
from .pii_redactor import PIIRedactor
from .pipeline import IngestionPipeline
from .quality_validator import QualityValidator

__all__ = [
    "DocumentProcessor",
    "IngestionPipeline",
    "LawsChunker",
    "NKPChunker",
    "PIIRedactor",
    "PgvectorIndexer",
    "QualityValidator",
]
