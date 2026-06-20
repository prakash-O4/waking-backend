"""
Document Processor for converting PDFs to clean Markdown
Uses Azure Document Intelligence for accurate Nepali text extraction
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from app.utils.pdf_to_markdown import PDFToMarkdownConverter
from app.rag_config import config
from app.utils.loggers import logger


@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    markdown_content: str
    source_file: str
    output_path: str
    metadata: Dict[str, Any]
    quality_score: float


class DocumentProcessor:
    """
    Processes PDF legal documents into clean, structured Markdown format
    with proper Nepali text handling
    """

    def __init__(self):
        """Initialize the document processor"""
        self.converter = PDFToMarkdownConverter(
            endpoint=config.ingestion.azure_endpoint,
            api_key=config.ingestion.azure_key
        )
        logger.info("DocumentProcessor initialized with Azure Document Intelligence")

    def process_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        namespace: str = "general"
    ) -> ProcessedDocument:
        """
        Process a PDF file into cleaned Markdown format

        Args:
            pdf_path: Path to the input PDF file
            output_path: Optional path for markdown output
            namespace: Document namespace/category

        Returns:
            ProcessedDocument with markdown content and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Generate output path if not provided
        if output_path is None:
            output_path = Path(config.ingestion.markdown_dir) / f"{pdf_file.stem}.md"

        # Convert PDF to Markdown using Azure
        try:
            markdown_content = self.converter.convert_pdf_to_markdown(
                str(pdf_path),
                str(output_path)
            )
        except Exception as e:
            logger.error(f"Azure conversion failed for {pdf_path}: {e}")
            raise

        # Clean and normalize the markdown content
        cleaned_content = self._clean_markdown(markdown_content)

        # Calculate quality score
        quality_score = self._calculate_quality_score(cleaned_content)

        # Extract basic metadata
        metadata = self._extract_basic_metadata(
            cleaned_content,
            str(pdf_path),
            namespace
        )

        processed_doc = ProcessedDocument(
            markdown_content=cleaned_content,
            source_file=str(pdf_path),
            output_path=str(output_path),
            metadata=metadata,
            quality_score=quality_score
        )

        logger.info(
            f"Document processed: {pdf_file.name} "
            f"(Quality: {quality_score:.2f}, "
            f"Length: {len(cleaned_content)} chars)"
        )

        return processed_doc

    def _clean_markdown(self, markdown_content: str) -> str:
        """
        Clean and normalize Markdown content for Nepali legal documents

        Args:
            markdown_content: Raw markdown content

        Returns:
            Cleaned markdown content
        """
        # Normalize Unicode (NFC normalization for Nepali text)
        content = unicodedata.normalize('NFC', markdown_content)

        # Remove zero-width characters that might interfere
        zero_width_chars = ['\u200B', '\u200C', '\u200D', '\u200E', '\u200F', '\uFEFF']
        for char in zero_width_chars:
            content = content.replace(char, '')

        # Fix spacing around Nepali punctuation
        # Add space after purna biram (।) if not present
        content = re.sub(r'।(?=[^\s।])', '। ', content)

        # Add space after double danda (॥) if not present
        content = re.sub(r'॥(?=[^\s॥])', '॥ ', content)

        # Remove excessive whitespace
        content = re.sub(r' +', ' ', content)  # Multiple spaces to single
        content = re.sub(r'\n{4,}', '\n\n\n', content)  # Max 3 newlines

        # Clean up broken Devanagari conjuncts (if any)
        # This handles cases where virama (्) got separated
        content = re.sub(r'([क-ह])\s+्', r'\1्', content)

        # Remove the generic "Document Analysis" header added by Azure
        content = re.sub(
            r'^#\s*Document Analysis\s*\n\*Extracted using.*?\*\s*\n---\s*\n+',
            '',
            content,
            flags=re.MULTILINE
        )

        # Remove website watermarks (lawcommission.gov.np)
        content = re.sub(
            r'(?i)(www\.)?lawcommission\.gov\.np\s*',
            '',
            content
        )

        # Remove page numbers and headers/footers
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(
            r'^(पृष्ठ|page|Page)\s*[०-९\d]+\s*$',
            '',
            content,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # Clean up gazette references if they appear alone
        content = re.sub(
            r'^नेपाल\s+राजपत्र\s*$',
            '',
            content,
            flags=re.MULTILINE
        )

        # Remove duplicate blank lines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Trim whitespace from each line
        lines = [line.rstrip() for line in content.split('\n')]
        content = '\n'.join(lines)

        return content.strip()

    def _calculate_quality_score(self, content: str) -> float:
        """
        Calculate a quality score for the extracted content

        Args:
            content: Markdown content

        Returns:
            Quality score between 0 and 1
        """
        if not content:
            return 0.0

        score = 0.0
        max_score = 100.0

        # Check 1: Has reasonable length (20 points)
        if len(content) > 500:
            score += 20
        elif len(content) > 100:
            score += 10

        # Check 2: Contains Nepali characters (30 points)
        nepali_chars = re.findall(r'[\u0900-\u097F]', content)
        if len(nepali_chars) > 100:
            score += 30
        elif len(nepali_chars) > 20:
            score += 15

        # Check 3: Has proper structure with headers (20 points)
        headers = re.findall(r'^#{1,3}\s+', content, re.MULTILINE)
        if len(headers) >= 5:
            score += 20
        elif len(headers) >= 2:
            score += 10

        # Check 4: Contains legal structure markers (15 points)
        legal_markers = [
            r'दफा\s*[०-९\d]+',  # Dafa (section)
            r'परिच्छेद',         # Parichhed (chapter)
            r'भाग',             # Bhag (part)
        ]
        for pattern in legal_markers:
            if re.search(pattern, content):
                score += 5

        # Check 5: No excessive garbage characters (15 points)
        # Check ratio of alphanumeric to special chars
        alphanumeric = len(re.findall(r'[\u0900-\u097Fa-zA-Z0-9]', content))
        total_chars = len(content.replace(' ', '').replace('\n', ''))
        if total_chars > 0:
            ratio = alphanumeric / total_chars
            if ratio > 0.8:
                score += 15
            elif ratio > 0.6:
                score += 10

        return min(score / max_score, 1.0)

    def _extract_basic_metadata(
        self,
        content: str,
        source_file: str,
        namespace: str
    ) -> Dict[str, Any]:
        """
        Extract basic metadata from markdown content

        Args:
            content: Markdown content
            source_file: Source PDF file path
            namespace: Document namespace

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "source_file": source_file,
            "namespace": namespace,
            "file_name": Path(source_file).name,
        }

        # Try to extract law name (first significant heading or title)
        law_name_match = re.search(
            r'([^\n]+(?:ऐन|संहिता|नियमावली|विधान)[^\n]*)',
            content
        )
        if law_name_match:
            metadata["law_name"] = law_name_match.group(1).strip()

        # Try to extract year
        year_match = re.search(r'[२०]{2}[०-९]{2}', content)
        if year_match:
            metadata["year"] = year_match.group(0)

        # Count sections, chapters, parts
        metadata["num_sections"] = len(re.findall(r'###\s+दफा', content))
        metadata["num_chapters"] = len(re.findall(r'##\s+परिच्छेद', content))
        metadata["num_parts"] = len(re.findall(r'#\s+भाग', content))

        # Document length metrics
        metadata["char_count"] = len(content)
        metadata["word_count"] = len(content.split())
        metadata["line_count"] = len(content.split('\n'))

        return metadata

    def batch_process(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        namespace: str = "general",
        pattern: str = "*.pdf"
    ) -> Dict[str, ProcessedDocument]:
        """
        Process multiple PDF files in batch

        Args:
            input_dir: Directory containing PDF files
            output_dir: Output directory for markdown files
            namespace: Document namespace
            pattern: Glob pattern for PDF files

        Returns:
            Dictionary mapping file names to ProcessedDocument objects
        """
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(config.ingestion.markdown_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_path.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = {}
        for idx, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")

            output_file = output_path / f"{pdf_file.stem}.md"

            try:
                processed_doc = self.process_pdf(
                    str(pdf_file),
                    str(output_file),
                    namespace
                )
                results[pdf_file.name] = processed_doc
                logger.info(f"✓ Successfully processed: {pdf_file.name}")

            except Exception as e:
                logger.error(f"✗ Error processing {pdf_file.name}: {e}")
                continue

        logger.info(
            f"Batch processing complete: {len(results)}/{len(pdf_files)} files processed"
        )
        return results
