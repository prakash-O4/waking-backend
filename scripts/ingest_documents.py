#!/usr/bin/env python3
"""
Complete Document Ingestion Pipeline for Nepali Legal Documents

This script demonstrates the full ingestion workflow:
1. PDF to Markdown conversion with Azure Document Intelligence
2. Hierarchical chunking (parent-child)
3. Metadata enrichment
4. Quality validation
5. Pinecone indexing with namespaces

Usage:
    python scripts/ingest_documents.py --source source/ --namespace general
    python scripts/ingest_documents.py --pdf source/Constitution.pdf --namespace constitution
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion import DocumentProcessor, QualityValidator

# PE-B NOTE: HierarchicalChunker / MetadataEnricher / PineconeIndexer were
# removed or rewritten in PE-A (branch pe-a/ingestion-pipeline). This PDF
# pipeline is PE-B scope; the guarded import keeps module startup working
# while its runtime path is reworked against the new ingestion stack.
try:
    from app.ingestion import (  # type: ignore[attr-defined]
        HierarchicalChunker,
        MetadataEnricher,
        PineconeIndexer,
    )
except ImportError:  # pragma: no cover
    HierarchicalChunker = None  # type: ignore[assignment]
    MetadataEnricher = None  # type: ignore[assignment]
    PineconeIndexer = None  # type: ignore[assignment]
from app.rag_config import config
from app.utils.loggers import logger


def ingest_single_document(
    pdf_path: str,
    namespace: str = "general",
    skip_validation: bool = False
) -> dict:
    """
    Ingest a single PDF document through the complete pipeline

    Args:
        pdf_path: Path to PDF file
        namespace: Pinecone namespace for organizing documents
        skip_validation: Skip quality validation

    Returns:
        Dictionary with ingestion statistics
    """
    logger.info(f"=" * 80)
    logger.info(f"Ingesting document: {pdf_path}")
    logger.info(f"Namespace: {namespace}")
    logger.info(f"=" * 80)

    stats = {
        "source_file": pdf_path,
        "namespace": namespace,
        "success": False,
        "error": None
    }

    try:
        # Step 1: Process PDF to Markdown
        logger.info("\n[Step 1/6] Processing PDF to Markdown...")
        processor = DocumentProcessor()
        processed_doc = processor.process_pdf(pdf_path, namespace=namespace)

        logger.info(f"✓ Document processed: Quality score = {processed_doc.quality_score:.2f}")
        logger.info(f"  - Output: {processed_doc.output_path}")
        logger.info(f"  - Length: {len(processed_doc.markdown_content)} characters")

        stats["markdown_output"] = processed_doc.output_path
        stats["quality_score"] = processed_doc.quality_score

        # Step 2: Create hierarchical chunks
        logger.info("\n[Step 2/6] Creating hierarchical chunks...")
        chunker = HierarchicalChunker()

        base_metadata = {
            **processed_doc.metadata,
            "quality_score": processed_doc.quality_score,
            "namespace": namespace
        }

        chunks = chunker.create_hierarchical_chunks(
            processed_doc.markdown_content,
            base_metadata
        )

        chunk_stats = chunker.get_statistics(chunks)
        logger.info(f"✓ Created {chunk_stats['total_chunks']} chunks")
        logger.info(f"  - Parents: {chunk_stats['parent_chunks']}")
        logger.info(f"  - Children: {chunk_stats['child_chunks']}")
        logger.info(f"  - Avg parent size: {chunk_stats['avg_parent_size_chars']:.0f} chars")
        logger.info(f"  - Avg child size: {chunk_stats['avg_child_size_chars']:.0f} chars")

        stats.update(chunk_stats)

        # Step 3: Enrich metadata
        logger.info("\n[Step 3/6] Enriching metadata...")
        enricher = MetadataEnricher(use_llm=True)

        # Convert chunks to dict format for enrichment
        chunks_dict = [
            {"content": c.content, "metadata": c.metadata}
            for c in chunks
        ]

        enriched_chunks = enricher.enrich_chunks(chunks_dict)

        # Convert back to HierarchicalChunk objects
        for i, chunk in enumerate(chunks):
            chunk.metadata = enriched_chunks[i]["metadata"]

        logger.info(f"✓ Metadata enriched for {len(chunks)} chunks")

        # Step 4: Validate quality
        if not skip_validation:
            logger.info("\n[Step 4/6] Validating chunk quality...")
            validator = QualityValidator()

            valid_chunks, validation_stats = validator.validate_chunks(
                chunks,
                strict=False,
                filter_invalid=True
            )

            logger.info(f"✓ Validation complete:")
            logger.info(f"  - Valid: {validation_stats['valid']}/{validation_stats['total']}")
            logger.info(f"  - Invalid: {validation_stats['invalid']}")
            logger.info(f"  - Avg score: {validation_stats['avg_score']:.2f}")

            if validation_stats['issues_count']:
                logger.warning(f"  Issues found: {validation_stats['issues_count']}")

            stats.update(validation_stats)
            chunks = valid_chunks
        else:
            logger.info("\n[Step 4/6] Skipping validation...")

        # Step 5: Save chunks to JSON
        logger.info("\n[Step 5/6] Saving chunks to JSON...")
        pdf_name = Path(pdf_path).stem
        chunks_output = Path(config.ingestion.chunks_dir) / f"{pdf_name}_chunks.json"
        chunks_output.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = [
            {
                "content": c.content,
                "metadata": c.metadata,
                "relationship": {
                    "chunk_id": c.relationship.chunk_id,
                    "parent_id": c.relationship.parent_id,
                    "child_ids": c.relationship.child_ids,
                    "chunk_type": c.relationship.chunk_type,
                    "level": c.relationship.level
                }
            }
            for c in chunks
        ]

        with open(chunks_output, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Chunks saved to: {chunks_output}")
        stats["chunks_output"] = str(chunks_output)

        # Step 6: Index to Pinecone
        logger.info("\n[Step 6/6] Indexing to Pinecone...")
        indexer = PineconeIndexer()

        indexing_stats = indexer.index_chunks(
            chunks,
            namespace=namespace
        )

        logger.info(f"✓ Indexing complete:")
        logger.info(f"  - Indexed: {indexing_stats['indexed']}")
        logger.info(f"  - Failed: {indexing_stats['failed']}")
        logger.info(f"  - Parent chunks: {indexing_stats['parent_chunks']}")
        logger.info(f"  - Child chunks: {indexing_stats['child_chunks']}")

        stats.update(indexing_stats)
        stats["success"] = True

    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}", exc_info=True)
        stats["error"] = str(e)
        stats["success"] = False

    return stats


def ingest_directory(
    source_dir: str,
    namespace: str = "general",
    skip_validation: bool = False
) -> dict:
    """
    Ingest all PDF files in a directory

    Args:
        source_dir: Directory containing PDF files
        namespace: Pinecone namespace
        skip_validation: Skip quality validation

    Returns:
        Dictionary with batch ingestion statistics
    """
    source_path = Path(source_dir)
    pdf_files = list(source_path.glob("*.pdf"))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Batch Ingestion")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info(f"Namespace: {namespace}")
    logger.info(f"{'=' * 80}\n")

    batch_stats = {
        "total_files": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "results": []
    }

    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")

        result = ingest_single_document(
            str(pdf_file),
            namespace=namespace,
            skip_validation=skip_validation
        )

        batch_stats["results"].append(result)

        if result["success"]:
            batch_stats["successful"] += 1
        else:
            batch_stats["failed"] += 1

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Batch Ingestion Complete")
    logger.info(f"Total files: {batch_stats['total_files']}")
    logger.info(f"Successful: {batch_stats['successful']}")
    logger.info(f"Failed: {batch_stats['failed']}")
    logger.info(f"{'=' * 80}")

    return batch_stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ingest Nepali legal documents into the RAG system"
    )

    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to a single PDF file to ingest"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="source",
        help="Directory containing PDF files (default: source/)"
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="general",
        choices=["general", "constitution", "criminal", "civil", "labor"],
        help="Pinecone namespace for organizing documents"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip quality validation step"
    )

    parser.add_argument(
        "--output-stats",
        type=str,
        help="Path to save ingestion statistics JSON"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure all required API keys are set")
        sys.exit(1)

    # Run ingestion
    if args.pdf:
        # Single file mode
        stats = ingest_single_document(
            args.pdf,
            namespace=args.namespace,
            skip_validation=args.skip_validation
        )
    else:
        # Directory mode
        stats = ingest_directory(
            args.source,
            namespace=args.namespace,
            skip_validation=args.skip_validation
        )

    # Save statistics if requested
    if args.output_stats:
        with open(args.output_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\nStatistics saved to: {args.output_stats}")

    # Exit code based on success
    if isinstance(stats, dict) and stats.get("success"):
        sys.exit(0)
    elif isinstance(stats, dict) and stats.get("failed", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
