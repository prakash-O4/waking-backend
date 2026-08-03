"""
Quality Validator for ensuring document and chunk quality before indexing
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from app.rag_config import config
from app.utils.loggers import logger

# HierarchicalChunk was removed with hierarchical_chunker.py in PE-A; the
# validator's runtime logic is unchanged and only duck-types chunk objects.
HierarchicalChunk = Any


@dataclass
class ValidationResult:
    """Result of quality validation"""
    is_valid: bool
    score: float
    issues: List[str]
    warnings: List[str]


class QualityValidator:
    """
    Validates document and chunk quality before indexing

    Checks for:
    - Proper Nepali text rendering
    - Minimum/maximum chunk lengths
    - Content quality (not just garbage/headers/footers)
    - Metadata completeness
    - Legal structure markers
    """

    def __init__(self):
        """Initialize the quality validator"""
        self.min_chunk_length = config.ingestion.min_chunk_length
        self.max_chunk_length = config.ingestion.max_chunk_length
        self.min_quality_score = config.ingestion.min_quality_score

        logger.info("QualityValidator initialized")

    def validate_chunk(
        self,
        chunk: HierarchicalChunk,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate a single hierarchical chunk

        Args:
            chunk: HierarchicalChunk to validate
            strict: If True, apply stricter validation rules

        Returns:
            ValidationResult with validation details
        """
        content = chunk.content
        metadata = chunk.metadata

        issues = []
        warnings = []
        scores = []

        # Check 1: Length validation
        length_score, length_issues = self._validate_length(content, strict)
        scores.append(length_score)
        issues.extend(length_issues)

        # Check 2: Nepali text validation
        nepali_score, nepali_issues = self._validate_nepali_text(content)
        scores.append(nepali_score)
        if nepali_score < 0.3:
            issues.extend(nepali_issues)
        else:
            warnings.extend(nepali_issues)

        # Check 3: Content quality (not just noise)
        content_score, content_issues = self._validate_content_quality(content)
        scores.append(content_score)
        if content_score < 0.5:
            issues.extend(content_issues)

        # Check 4: Metadata validation
        metadata_score, metadata_warnings = self._validate_metadata(metadata, strict)
        scores.append(metadata_score)
        warnings.extend(metadata_warnings)

        # Check 5: Legal structure markers (for legal documents)
        structure_score, structure_warnings = self._validate_legal_structure(content)
        scores.append(structure_score)
        warnings.extend(structure_warnings)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Determine if valid
        is_valid = (
            overall_score >= self.min_quality_score
            and len(issues) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            issues=issues,
            warnings=warnings
        )

    def validate_chunks(
        self,
        chunks: List[HierarchicalChunk],
        strict: bool = False,
        filter_invalid: bool = True
    ) -> Tuple[List[HierarchicalChunk], Dict[str, Any]]:
        """
        Validate multiple chunks

        Args:
            chunks: List of HierarchicalChunk objects
            strict: Apply strict validation
            filter_invalid: Remove invalid chunks from results

        Returns:
            Tuple of (valid_chunks, validation_stats)
        """
        logger.info(f"Validating {len(chunks)} chunks (strict={strict})")

        valid_chunks = []
        validation_stats = {
            "total": len(chunks),
            "valid": 0,
            "invalid": 0,
            "avg_score": 0.0,
            "issues_count": {},
            "warnings_count": {}
        }

        scores = []

        for chunk in chunks:
            result = self.validate_chunk(chunk, strict)
            scores.append(result.score)

            if result.is_valid:
                valid_chunks.append(chunk)
                validation_stats["valid"] += 1
            else:
                validation_stats["invalid"] += 1

                if not filter_invalid:
                    valid_chunks.append(chunk)

            # Count issues
            for issue in result.issues:
                issue_key = issue.split(":")[0] if ":" in issue else issue
                validation_stats["issues_count"][issue_key] = \
                    validation_stats["issues_count"].get(issue_key, 0) + 1

            # Count warnings
            for warning in result.warnings:
                warning_key = warning.split(":")[0] if ":" in warning else warning
                validation_stats["warnings_count"][warning_key] = \
                    validation_stats["warnings_count"].get(warning_key, 0) + 1

        validation_stats["avg_score"] = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            f"Validation complete: {validation_stats['valid']}/{validation_stats['total']} "
            f"chunks valid (avg score: {validation_stats['avg_score']:.2f})"
        )

        return valid_chunks, validation_stats

    def _validate_length(self, content: str, strict: bool) -> Tuple[float, List[str]]:
        """Validate chunk length"""
        issues = []
        length = len(content)

        if length < self.min_chunk_length:
            issues.append(f"Length too short: {length} < {self.min_chunk_length}")
            return 0.0, issues

        if length > self.max_chunk_length:
            issues.append(f"Length too long: {length} > {self.max_chunk_length}")
            return 0.0, issues

        # Score based on ideal length (800 chars for child, 2000 for parent)
        if strict:
            ideal_length = config.chunking.child_chunk_size
        else:
            ideal_length = config.chunking.parent_chunk_size

        # Calculate score (1.0 at ideal, decreasing as we move away)
        diff = abs(length - ideal_length)
        max_diff = ideal_length * 0.5  # Allow 50% deviation
        score = max(0.0, 1.0 - (diff / max_diff))

        return score, issues

    def _validate_nepali_text(self, content: str) -> Tuple[float, List[str]]:
        """Validate presence and quality of Nepali text"""
        issues = []

        # Count Nepali characters (Devanagari script)
        nepali_chars = len(re.findall(r'[\u0900-\u097F]', content))
        total_chars = len(re.sub(r'\s', '', content))  # Excluding whitespace

        if total_chars == 0:
            issues.append("Content empty: no characters")
            return 0.0, issues

        nepali_ratio = nepali_chars / total_chars if total_chars > 0 else 0

        # For legal documents, we expect significant Nepali content
        if nepali_ratio < 0.3:
            issues.append(f"Low Nepali content: {nepali_ratio:.1%} Nepali characters")

        # Check for common rendering issues
        if '\ufffd' in content:  # Replacement character (rendering error)
            issues.append("Text rendering error: contains replacement characters")
            return 0.0, issues

        # Score is the ratio of Nepali characters
        return min(nepali_ratio * 2, 1.0), issues

    def _validate_content_quality(self, content: str) -> Tuple[float, List[str]]:
        """Validate that content is meaningful (not just headers/footers)"""
        issues = []

        # Check if content is mostly just website URLs or page numbers
        noise_patterns = [
            r'(www\.)?lawcommission\.gov\.np',
            r'^(पृष्ठ|page)\s*\d+',
            r'^\d+\s*$',
        ]

        noise_ratio = 0.0
        for pattern in noise_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            noise_ratio += len(''.join(matches)) / len(content) if content else 0

        if noise_ratio > 0.5:
            issues.append(f"High noise content: {noise_ratio:.1%} appears to be headers/footers")
            return 0.0, issues

        # Check for reasonable word count
        words = content.split()
        if len(words) < 10:
            issues.append(f"Too few words: {len(words)} words")
            return 0.3, issues

        # Check for excessive repeated characters (garbage text)
        max_repeats = max(
            (len(list(g)) for char, g in __import__('itertools').groupby(content)),
            default=0
        )
        if max_repeats > 20:
            issues.append(f"Excessive character repetition: {max_repeats} consecutive characters")
            return 0.2, issues

        # Score based on word density and variety
        unique_words = len(set(words))
        word_variety = unique_words / len(words) if words else 0

        score = word_variety * 0.7 + (1 - noise_ratio) * 0.3

        return score, issues

    def _validate_metadata(
        self,
        metadata: Dict[str, Any],
        strict: bool
    ) -> Tuple[float, List[str]]:
        """Validate metadata completeness"""
        warnings = []

        required_fields = ["chunk_id", "chunk_type"]
        recommended_fields = ["source_file", "namespace", "char_count"]

        # Check required fields
        missing_required = [f for f in required_fields if f not in metadata]
        if missing_required:
            warnings.append(f"Missing required metadata: {', '.join(missing_required)}")
            return 0.0, warnings

        # Check recommended fields
        missing_recommended = [f for f in recommended_fields if f not in metadata]
        if missing_recommended and strict:
            warnings.append(f"Missing recommended metadata: {', '.join(missing_recommended)}")

        # Score based on metadata completeness
        total_fields = len(required_fields) + len(recommended_fields)
        present_fields = len(required_fields) + (len(recommended_fields) - len(missing_recommended))
        score = present_fields / total_fields

        return score, warnings

    def _validate_legal_structure(self, content: str) -> Tuple[float, List[str]]:
        """Validate presence of legal structure markers"""
        warnings = []

        # Check for legal structure markers
        markers = {
            "section": r'दफा\s*[०-९\d]+',
            "chapter": r'परिच्छेद',
            "part": r'भाग',
            "subsection": r'\([क-ज्ञ०-९a-z]+\)',
        }

        found_markers = []
        for marker_name, pattern in markers.items():
            if re.search(pattern, content):
                found_markers.append(marker_name)

        if not found_markers:
            warnings.append("No legal structure markers found (may not be a legal section)")
            return 0.7, warnings  # Not critical, but worth noting

        # Score based on number of markers found
        score = len(found_markers) / len(markers)

        return score, warnings

    def generate_report(
        self,
        validation_stats: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable validation report

        Args:
            validation_stats: Stats from validate_chunks()

        Returns:
            Formatted report string
        """
        report = ["Quality Validation Report", "=" * 50]

        report.append(f"\nTotal chunks: {validation_stats['total']}")
        report.append(f"Valid: {validation_stats['valid']}")
        report.append(f"Invalid: {validation_stats['invalid']}")
        report.append(f"Average score: {validation_stats['avg_score']:.2f}")

        if validation_stats['issues_count']:
            report.append("\nIssues found:")
            for issue, count in validation_stats['issues_count'].items():
                report.append(f"  - {issue}: {count}")

        if validation_stats['warnings_count']:
            report.append("\nWarnings:")
            for warning, count in validation_stats['warnings_count'].items():
                report.append(f"  - {warning}: {count}")

        return "\n".join(report)
