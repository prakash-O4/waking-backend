"""
Metadata Enricher for extracting and enhancing legal document metadata
Handles Nepali legal document structure and terminology
"""

import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from app.rag_config import config
from app.utils.loggers import logger


class MetadataEnricher:
    """
    Enriches chunks with detailed legal metadata including:
    - Law name (Nepali & English)
    - Legal hierarchy (Part, Chapter, Section)
    - Enactment dates and amendments
    - Keywords and legal terms
    - Cross-references
    """

    # Regex patterns for legal structure extraction
    PATTERNS = {
        # Law names
        "law_name_with_year": r"([\u0900-\u097F\s]+(?:संहिता|ऐन|नियमावली|विधान)[,\s]*[२०]{2}[०-९]{2})",
        "law_name": r"([\u0900-\u097F\s]{10,}(?:संहिता|ऐन|नियमावली|विधान))",

        # Structure markers
        "part": r"भाग[-\s]*([०-९\d]+)",
        "chapter": r"परिच्छेद[-\s]*([०-९\d]+)",
        "section": r"दफा\s*([०-९\d]+)",
        "sub_section": r"\(([क-ज्ञ०-९a-z\d]+)\)",

        # Dates
        "nepali_date": r"[२०]{2}[०-९]{2}[।\.][०-१]{2}[।\.][०-३]{2}",
        "year": r"[२०]{2}[०-९]{2}",

        # References
        "dafa_reference": r"दफा\s*([०-९\d]+)",
        "article_reference": r"धारा\s*([०-९\d]+)",
    }

    # Legal keywords (Nepali)
    LEGAL_KEYWORDS = [
        "अधिकार", "कर्तव्य", "दायित्व", "सजाय", "जरिबाना",
        "न्यायालय", "अदालत", "मुद्दा", "उजुरी", "अपील",
        "संविधान", "कानून", "ऐन", "नियम", "विनियम",
        "सरकार", "नागरिक", "राज्य", "प्रदेश", "स्थानीय",
    ]

    def __init__(self, use_llm: bool = True):
        """
        Initialize metadata enricher

        Args:
            use_llm: Whether to use LLM for advanced metadata extraction
        """
        self.use_llm = use_llm

        if self.use_llm:
            self.llm = ChatOpenAI(
                model=config.generation.model,
                temperature=0,
                openai_api_key=config.openai_api_key
            )
            logger.info("MetadataEnricher initialized with LLM support")
        else:
            logger.info("MetadataEnricher initialized (regex-only mode)")

    def enrich_chunk(
        self,
        content: str,
        existing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a single chunk with enhanced metadata

        Args:
            content: Chunk content
            existing_metadata: Existing metadata dict

        Returns:
            Enhanced metadata dict
        """
        enriched = existing_metadata.copy()

        # Extract legal structure
        legal_structure = self._extract_legal_structure(content)
        enriched.update(legal_structure)

        # Extract keywords
        keywords = self._extract_keywords(content)
        enriched["keywords"] = keywords

        # Extract references
        references = self._extract_references(content)
        enriched["references"] = references

        # Extract dates if present
        dates = self._extract_dates(content)
        if dates:
            enriched["dates"] = dates

        # Add content statistics
        enriched["content_stats"] = self._get_content_stats(content)

        # LLM-based enrichment (if enabled)
        if self.use_llm and len(content) > 100:
            try:
                llm_metadata = self._llm_extract_metadata(content)
                enriched["llm_metadata"] = llm_metadata
            except Exception as e:
                logger.warning(f"LLM metadata extraction failed: {e}")

        return enriched

    def enrich_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich multiple chunks with metadata

        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata' keys

        Returns:
            List of chunks with enriched metadata
        """
        logger.info(f"Enriching metadata for {len(chunks)} chunks")

        enriched_chunks = []
        for idx, chunk in enumerate(chunks):
            if idx % 10 == 0:
                logger.info(f"Enriching chunk {idx}/{len(chunks)}")

            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            enriched_metadata = self.enrich_chunk(content, metadata)

            enriched_chunks.append({
                "content": content,
                "metadata": enriched_metadata
            })

        logger.info("Metadata enrichment complete")
        return enriched_chunks

    def _extract_legal_structure(self, content: str) -> Dict[str, Any]:
        """Extract legal structure information (Part, Chapter, Section, etc.)"""
        structure = {}

        # Extract Part (भाग)
        part_match = re.search(self.PATTERNS["part"], content)
        if part_match:
            structure["part"] = part_match.group(1)

        # Extract Chapter (परिच्छेद)
        chapter_match = re.search(self.PATTERNS["chapter"], content)
        if chapter_match:
            structure["chapter"] = chapter_match.group(1)

        # Extract Section (दफा)
        section_match = re.search(self.PATTERNS["section"], content)
        if section_match:
            structure["section"] = section_match.group(1)

        # Extract sub-sections
        subsection_matches = re.findall(self.PATTERNS["sub_section"], content)
        if subsection_matches:
            structure["subsections"] = subsection_matches[:5]  # Limit to first 5

        # Try to extract law name if not already present
        if "law_name" not in structure:
            law_name_match = re.search(self.PATTERNS["law_name_with_year"], content)
            if not law_name_match:
                law_name_match = re.search(self.PATTERNS["law_name"], content)

            if law_name_match:
                structure["law_name"] = law_name_match.group(1).strip()

        return structure

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract legal keywords from content"""
        keywords = []

        # Check for predefined legal keywords
        for keyword in self.LEGAL_KEYWORDS:
            if keyword in content:
                keywords.append(keyword)

        # Limit to top 10 keywords
        return keywords[:10]

    def _extract_references(self, content: str) -> Dict[str, List[str]]:
        """Extract cross-references to other sections/articles"""
        references = {}

        # Extract Dafa references
        dafa_refs = re.findall(self.PATTERNS["dafa_reference"], content)
        if dafa_refs:
            references["dafa"] = list(set(dafa_refs))[:5]  # Unique, max 5

        # Extract Article references
        article_refs = re.findall(self.PATTERNS["article_reference"], content)
        if article_refs:
            references["article"] = list(set(article_refs))[:5]

        return references

    def _extract_dates(self, content: str) -> List[str]:
        """Extract Nepali dates from content"""
        # Extract full Nepali dates (YYYY.MM.DD format)
        dates = re.findall(self.PATTERNS["nepali_date"], content)

        # Also extract just years
        years = re.findall(self.PATTERNS["year"], content)

        all_dates = list(set(dates + years))
        return all_dates[:3]  # Max 3 dates

    def _get_content_stats(self, content: str) -> Dict[str, int]:
        """Get statistics about the content"""
        return {
            "char_count": len(content),
            "word_count": len(content.split()),
            "nepali_char_count": len(re.findall(r'[\u0900-\u097F]', content)),
            "sentence_count": len(re.split(r'[।॥\.\!]', content)),
        }

    def _llm_extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Use LLM to extract advanced metadata from content

        Args:
            content: Chunk content (first 1000 chars)

        Returns:
            Dictionary with LLM-extracted metadata
        """
        # Limit content to avoid token limits
        preview = content[:1000] if len(content) > 1000 else content

        prompt = f"""Extract metadata from this Nepali legal document excerpt.

Document excerpt:
{preview}

Extract:
1. Main topic/subject (in English)
2. Law type (constitution/criminal/civil/labor/other)
3. Key legal concepts mentioned (max 3)

Return ONLY valid JSON in this format:
{{
  "topic": "string",
  "law_type": "string",
  "legal_concepts": ["concept1", "concept2", "concept3"]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content_str = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            metadata = json.loads(content_str)
            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"LLM metadata extraction error: {e}")
            return {}

    def extract_law_info(self, markdown_content: str) -> Dict[str, Any]:
        """
        Extract high-level law information from full markdown content

        Args:
            markdown_content: Full markdown document

        Returns:
            Dictionary with law-level metadata
        """
        law_info = {}

        # Get first 2000 characters for law name extraction
        preview = markdown_content[:2000]

        # Extract law name
        law_name_match = re.search(self.PATTERNS["law_name_with_year"], preview)
        if not law_name_match:
            law_name_match = re.search(self.PATTERNS["law_name"], preview)

        if law_name_match:
            law_info["law_name"] = law_name_match.group(1).strip()

        # Extract year
        year_match = re.search(self.PATTERNS["year"], preview)
        if year_match:
            law_info["year"] = year_match.group(0)

        # Count structural elements
        law_info["total_parts"] = len(re.findall(self.PATTERNS["part"], markdown_content))
        law_info["total_chapters"] = len(re.findall(self.PATTERNS["chapter"], markdown_content))
        law_info["total_sections"] = len(re.findall(self.PATTERNS["section"], markdown_content))

        # Determine law type based on name
        law_name = law_info.get("law_name", "").lower()
        if "संविधान" in law_name or "constitution" in law_name:
            law_info["law_type"] = "constitution"
        elif "अपराध" in law_name or "criminal" in law_name:
            law_info["law_type"] = "criminal"
        elif "देवानी" in law_name or "civil" in law_name:
            law_info["law_type"] = "civil"
        elif "श्रम" in law_name or "labor" in law_name:
            law_info["law_type"] = "labor"
        else:
            law_info["law_type"] = "general"

        return law_info

    def create_citation_text(self, metadata: Dict[str, Any]) -> str:
        """
        Create a citation text from metadata

        Args:
            metadata: Chunk metadata

        Returns:
            Formatted citation string
        """
        parts = []

        # Add law name if present
        if "law_name" in metadata:
            parts.append(metadata["law_name"])

        # Add section if present
        if "section" in metadata:
            parts.append(f"दफा {metadata['section']}")
        elif "Section" in metadata:
            parts.append(metadata["Section"])

        # Add chapter if present
        if "chapter" in metadata and "section" not in metadata:
            parts.append(f"परिच्छेद {metadata['chapter']}")
        elif "Chapter" in metadata and "section" not in metadata:
            parts.append(metadata["Chapter"])

        # Add part if present
        if "part" in metadata and "chapter" not in metadata:
            parts.append(f"भाग {metadata['part']}")
        elif "Part" in metadata and "chapter" not in metadata:
            parts.append(metadata["Part"])

        return ", ".join(parts) if parts else "Unknown source"
