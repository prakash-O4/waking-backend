"""
Hierarchical Chunker implementing parent-child chunking strategy
for legal documents with semantic awareness
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document

from app.rag_config import config
from app.utils.loggers import logger


@dataclass
class ChunkRelationship:
    """Represents the hierarchical relationship between chunks"""
    chunk_id: str
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    chunk_type: str = "child"  # "parent" or "child"
    level: int = 0  # Hierarchical level (0=parent, 1=child, etc.)


@dataclass
class HierarchicalChunk:
    """Container for a chunk with hierarchical metadata"""
    content: str
    metadata: Dict[str, Any]
    relationship: ChunkRelationship

    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        # Merge relationship into metadata
        full_metadata = {
            **self.metadata,
            "chunk_id": self.relationship.chunk_id,
            "parent_id": self.relationship.parent_id,
            "child_ids": self.relationship.child_ids,
            "chunk_type": self.relationship.chunk_type,
            "level": self.relationship.level,
        }
        return Document(page_content=self.content, metadata=full_metadata)


class HierarchicalChunker:
    """
    Creates hierarchical parent-child chunks from Markdown documents
    Optimized for Nepali legal documents with semantic structure preservation
    """

    def __init__(self):
        """Initialize the hierarchical chunker"""
        self.config = config.chunking

        # Markdown header splitter for structure-aware splitting
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Part"),        # भाग
                ("##", "Chapter"),    # परिच्छेद
                ("###", "Section"),   # दफा
            ],
            strip_headers=False
        )

        # Parent chunk splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            separators=self.config.separators,
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=self.config.parent_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Child chunk splitter (smaller chunks for precise retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            separators=self.config.separators,
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.child_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info("HierarchicalChunker initialized")
        logger.info(f"Parent chunk size: {self.config.parent_chunk_size}")
        logger.info(f"Child chunk size: {self.config.child_chunk_size}")

    def create_hierarchical_chunks(
        self,
        markdown_content: str,
        base_metadata: Dict[str, Any]
    ) -> List[HierarchicalChunk]:
        """
        Create hierarchical parent-child chunks from markdown content

        Args:
            markdown_content: Markdown formatted document
            base_metadata: Base metadata to attach to all chunks

        Returns:
            List of HierarchicalChunk objects (both parents and children)
        """
        logger.info("Creating hierarchical chunks from markdown")

        all_chunks = []

        # Step 1: Split by headers to preserve semantic structure
        try:
            header_splits = self.header_splitter.split_text(markdown_content)
        except Exception as e:
            logger.warning(f"Header splitting failed, using fallback: {e}")
            # Fallback: treat entire content as one section
            header_splits = [
                Document(page_content=markdown_content, metadata={})
            ]

        logger.info(f"Created {len(header_splits)} header-based sections")

        # Step 2: For each section, create parent and child chunks
        for section_idx, section_doc in enumerate(header_splits):
            section_content = section_doc.page_content
            section_metadata = {**base_metadata, **section_doc.metadata}

            # Create parent chunks for this section
            parent_chunks = self._create_parent_chunks(
                section_content,
                section_metadata,
                section_idx
            )

            # For each parent, create child chunks
            for parent_chunk in parent_chunks:
                # Create children from this parent
                child_chunks = self._create_child_chunks(
                    parent_chunk.content,
                    parent_chunk.metadata,
                    parent_chunk.relationship.chunk_id
                )

                # Update parent's child_ids
                parent_chunk.relationship.child_ids = [
                    child.relationship.chunk_id for child in child_chunks
                ]

                # Add parent and all its children to results
                all_chunks.append(parent_chunk)
                all_chunks.extend(child_chunks)

        logger.info(
            f"Created {len(all_chunks)} total chunks "
            f"(parents: {sum(1 for c in all_chunks if c.relationship.chunk_type == 'parent')}, "
            f"children: {sum(1 for c in all_chunks if c.relationship.chunk_type == 'child')})"
        )

        return all_chunks

    def _create_parent_chunks(
        self,
        content: str,
        metadata: Dict[str, Any],
        section_idx: int
    ) -> List[HierarchicalChunk]:
        """
        Create parent chunks from content

        Args:
            content: Section content
            metadata: Section metadata
            section_idx: Section index

        Returns:
            List of parent HierarchicalChunk objects
        """
        # Split into parent-sized chunks
        parent_texts = self.parent_splitter.split_text(content)

        parent_chunks = []
        for idx, parent_text in enumerate(parent_texts):
            chunk_id = str(uuid.uuid4())

            parent_metadata = {
                **metadata,
                "section_index": section_idx,
                "parent_index": idx,
                "char_count": len(parent_text),
                "word_count": len(parent_text.split()),
            }

            relationship = ChunkRelationship(
                chunk_id=chunk_id,
                parent_id=None,  # Parents have no parent
                child_ids=[],    # Will be filled later
                chunk_type="parent",
                level=0
            )

            parent_chunk = HierarchicalChunk(
                content=parent_text,
                metadata=parent_metadata,
                relationship=relationship
            )

            parent_chunks.append(parent_chunk)

        return parent_chunks

    def _create_child_chunks(
        self,
        parent_content: str,
        parent_metadata: Dict[str, Any],
        parent_id: str
    ) -> List[HierarchicalChunk]:
        """
        Create child chunks from a parent chunk

        Args:
            parent_content: Parent chunk content
            parent_metadata: Parent chunk metadata
            parent_id: Parent chunk ID

        Returns:
            List of child HierarchicalChunk objects
        """
        # Split parent into child-sized chunks
        child_texts = self.child_splitter.split_text(parent_content)

        child_chunks = []
        for idx, child_text in enumerate(child_texts):
            chunk_id = str(uuid.uuid4())

            child_metadata = {
                **parent_metadata,
                "child_index": idx,
                "total_children": len(child_texts),
                "char_count": len(child_text),
                "word_count": len(child_text.split()),
            }

            relationship = ChunkRelationship(
                chunk_id=chunk_id,
                parent_id=parent_id,
                child_ids=[],  # Children don't have children
                chunk_type="child",
                level=1
            )

            child_chunk = HierarchicalChunk(
                content=child_text,
                metadata=child_metadata,
                relationship=relationship
            )

            child_chunks.append(child_chunk)

        return child_chunks

    def get_parent_chunks(
        self,
        all_chunks: List[HierarchicalChunk]
    ) -> List[HierarchicalChunk]:
        """Extract only parent chunks from a list of hierarchical chunks"""
        return [c for c in all_chunks if c.relationship.chunk_type == "parent"]

    def get_child_chunks(
        self,
        all_chunks: List[HierarchicalChunk]
    ) -> List[HierarchicalChunk]:
        """Extract only child chunks from a list of hierarchical chunks"""
        return [c for c in all_chunks if c.relationship.chunk_type == "child"]

    def get_parent_for_child(
        self,
        child_chunk: HierarchicalChunk,
        all_chunks: List[HierarchicalChunk]
    ) -> Optional[HierarchicalChunk]:
        """
        Find the parent chunk for a given child chunk

        Args:
            child_chunk: The child chunk
            all_chunks: List of all chunks to search in

        Returns:
            Parent chunk if found, None otherwise
        """
        if child_chunk.relationship.parent_id is None:
            return None

        for chunk in all_chunks:
            if chunk.relationship.chunk_id == child_chunk.relationship.parent_id:
                return chunk

        return None

    def get_children_for_parent(
        self,
        parent_chunk: HierarchicalChunk,
        all_chunks: List[HierarchicalChunk]
    ) -> List[HierarchicalChunk]:
        """
        Find all child chunks for a given parent chunk

        Args:
            parent_chunk: The parent chunk
            all_chunks: List of all chunks to search in

        Returns:
            List of child chunks
        """
        child_ids = set(parent_chunk.relationship.child_ids)
        return [
            chunk for chunk in all_chunks
            if chunk.relationship.chunk_id in child_ids
        ]

    def to_documents(
        self,
        chunks: List[HierarchicalChunk]
    ) -> List[Document]:
        """
        Convert list of HierarchicalChunks to LangChain Documents

        Args:
            chunks: List of HierarchicalChunk objects

        Returns:
            List of LangChain Document objects
        """
        return [chunk.to_document() for chunk in chunks]

    def get_statistics(
        self,
        chunks: List[HierarchicalChunk]
    ) -> Dict[str, Any]:
        """
        Get statistics about the chunking process

        Args:
            chunks: List of hierarchical chunks

        Returns:
            Dictionary with chunking statistics
        """
        parents = self.get_parent_chunks(chunks)
        children = self.get_child_chunks(chunks)

        if not chunks:
            return {"error": "No chunks provided"}

        # Calculate average sizes
        avg_parent_size = (
            sum(len(p.content) for p in parents) / len(parents)
            if parents else 0
        )
        avg_child_size = (
            sum(len(c.content) for c in children) / len(children)
            if children else 0
        )

        # Calculate children per parent
        children_per_parent = (
            len(children) / len(parents)
            if parents else 0
        )

        return {
            "total_chunks": len(chunks),
            "parent_chunks": len(parents),
            "child_chunks": len(children),
            "avg_parent_size_chars": avg_parent_size,
            "avg_child_size_chars": avg_child_size,
            "avg_children_per_parent": children_per_parent,
            "min_chunk_size": min(len(c.content) for c in chunks),
            "max_chunk_size": max(len(c.content) for c in chunks),
        }
