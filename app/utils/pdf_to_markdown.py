"""
PDF to Markdown Converter using Azure AI Document Intelligence
Extracts text and tables from PDF files while preserving Nepali characters
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from project root
# This finds the .env file regardless of where the script is run from
project_root = Path(__file__).parent.parent.parent  # Go up to project root
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


class PDFToMarkdownConverter:
    """
    Converts PDF documents to structured Markdown format using Azure Document Intelligence.
    Supports Nepali language and preserves Unicode characters.
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the converter with Azure credentials.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key: Azure Document Intelligence API key
        """
        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Document Intelligence credentials not found. "
                "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables."
            )

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )

    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert a PDF file to Markdown format.

        Args:
            pdf_path: Path to the input PDF file
            output_path: Optional path to save the output Markdown file

        Returns:
            Markdown formatted string
        """
        print(f"Processing PDF: {pdf_path}")

        # Read PDF file
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()

        # Analyze document using Azure Document Intelligence
        print("Analyzing document with Azure AI...")
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-layout",  # Use layout model for text and tables
            analyze_request=pdf_content,
            content_type="application/pdf",
            locale="ne"  # Nepali language code
        )

        result = poller.result()

        # Convert to Markdown
        markdown_content = self._convert_to_markdown(result)

        # Save to file if output path is provided
        if output_path:
            self._save_markdown(markdown_content, output_path)
            print(f"Markdown saved to: {output_path}")

        return markdown_content

    def _convert_to_markdown(self, result) -> str:
        """
        Convert Azure Document Intelligence result to Markdown format.

        Args:
            result: Analysis result from Azure Document Intelligence

        Returns:
            Markdown formatted string
        """
        markdown_lines = []

        # Add document metadata
        markdown_lines.append(f"# Document Analysis\n")
        markdown_lines.append(f"*Extracted using Azure Document Intelligence*\n")
        markdown_lines.append("---\n")

        # Process pages
        if result.pages:
            for page_idx, page in enumerate(result.pages, 1):
                markdown_lines.append(f"\n## Page {page_idx}\n")

                # Track which content has been processed (to avoid duplication)
                processed_spans = set()

                # First, add tables
                if result.tables:
                    page_tables = [
                        table for table in result.tables
                        if self._is_on_page(table, page_idx)
                    ]

                    for table in page_tables:
                        markdown_table = self._format_table(table)
                        markdown_lines.append(markdown_table)
                        markdown_lines.append("\n")

                        # Mark table spans as processed
                        for cell in table.cells:
                            if cell.spans:
                                for span in cell.spans:
                                    processed_spans.add((span.offset, span.length))

                # Then add paragraphs (excluding table content)
                if result.paragraphs:
                    page_paragraphs = [
                        para for para in result.paragraphs
                        if self._is_on_page(para, page_idx)
                    ]

                    for para in page_paragraphs:
                        # Skip if this paragraph is part of a table
                        if self._is_in_processed_spans(para, processed_spans):
                            continue

                        # Format based on role
                        formatted_para = self._format_paragraph(para)
                        markdown_lines.append(formatted_para)
                        markdown_lines.append("\n")

        # If no pages found, fall back to raw text
        if not markdown_lines or len(markdown_lines) <= 3:
            if result.content:
                markdown_lines.append("\n## Extracted Content\n")
                markdown_lines.append(result.content)

        return "".join(markdown_lines)

    def _is_on_page(self, element, page_number: int) -> bool:
        """Check if an element belongs to a specific page."""
        if hasattr(element, 'bounding_regions') and element.bounding_regions:
            return any(
                region.page_number == page_number
                for region in element.bounding_regions
            )
        return False

    def _is_in_processed_spans(self, element, processed_spans: set) -> bool:
        """Check if element's content has already been processed."""
        if hasattr(element, 'spans') and element.spans:
            for span in element.spans:
                if (span.offset, span.length) in processed_spans:
                    return True
        return False

    def _format_paragraph(self, paragraph) -> str:
        """Format a paragraph based on its role."""
        content = paragraph.content.strip()

        if not content:
            return ""

        # Check paragraph role
        role = getattr(paragraph, 'role', None)

        if role == 'title':
            return f"# {content}\n"
        elif role == 'sectionHeading':
            return f"## {content}\n"
        elif role == 'pageHeader':
            return f"*{content}*\n"
        elif role == 'pageFooter':
            return f"*{content}*\n"
        elif role == 'footnote':
            return f"> {content}\n"
        else:
            return f"{content}\n"

    def _format_table(self, table) -> str:
        """Format a table to Markdown table syntax."""
        if not table.cells:
            return ""

        # Determine table dimensions
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1

        # Create empty grid
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]

        # Fill grid with cell content
        for cell in table.cells:
            row = cell.row_index
            col = cell.column_index
            content = cell.content.strip() if cell.content else ""
            grid[row][col] = content

        # Convert to Markdown table
        markdown_table = []

        for row_idx, row in enumerate(grid):
            # Escape pipe characters in cell content
            escaped_row = [cell.replace("|", "\\|") for cell in row]
            markdown_table.append("| " + " | ".join(escaped_row) + " |")

            # Add separator after first row (header)
            if row_idx == 0:
                markdown_table.append("| " + " | ".join(["---"] * max_col) + " |")

        return "\n".join(markdown_table)

    def _save_markdown(self, content: str, output_path: str) -> None:
        """Save Markdown content to a file with UTF-8 encoding."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.pdf"
    ) -> List[str]:
        """
        Convert multiple PDF files to Markdown.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save Markdown files
            pattern: Glob pattern for PDF files (default: "*.pdf")

        Returns:
            List of output file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_path.glob(pattern))
        output_files = []

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")

            # Generate output filename
            output_file = output_path / f"{pdf_file.stem}.md"

            try:
                self.convert_pdf_to_markdown(str(pdf_file), str(output_file))
                output_files.append(str(output_file))
                print(f"✓ Successfully converted: {pdf_file.name}")
            except Exception as e:
                print(f"✗ Error converting {pdf_file.name}: {str(e)}")

        print(f"\n\nCompleted: {len(output_files)}/{len(pdf_files)} files converted")
        return output_files


def main():
    """Example usage of the PDF to Markdown converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using Azure Document Intelligence"
    )
    parser.add_argument(
        "input",
        help="Input PDF file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file or directory"
    )
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Batch process all PDFs in input directory"
    )

    args = parser.parse_args()

    # Initialize converter
    converter = PDFToMarkdownConverter()

    if args.batch:
        # Batch processing
        output_dir = args.output or "./output"
        converter.batch_convert(args.input, output_dir)
    else:
        # Single file processing
        output_path = args.output
        if not output_path:
            input_path = Path(args.input)
            output_path = f"{input_path.stem}.md"

        markdown = converter.convert_pdf_to_markdown(args.input, output_path)
        print("\nConversion completed!")
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
