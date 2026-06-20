"""
Enhanced Pinecone Indexer with namespace support and hierarchical chunk indexing
"""

import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from app.rag_config import config
from app.utils.loggers import logger
from app.ingestion.hierarchical_chunker import HierarchicalChunk


class PineconeIndexer:
    """
    Enhanced Pinecone indexer with:
    - Namespace support for organizing documents
    - Hierarchical chunk handling (parent-child)
    - Batch processing with resume capability
    - Error handling and retries
    """

    def __init__(self):
        """Initialize the Pinecone indexer"""
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_key=config.openai_api_key
        )

        self.index_name = config.pinecone.index_name
        self.use_namespaces = config.pinecone.use_namespaces

        logger.info(f"PineconeIndexer initialized for index: {self.index_name}")

    def index_chunks(
        self,
        chunks: List[HierarchicalChunk],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Index hierarchical chunks to Pinecone

        Args:
            chunks: List of HierarchicalChunk objects
            namespace: Optional namespace for organizing documents
            batch_size: Batch size for processing (default from config)

        Returns:
            Dictionary with indexing statistics
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return {"indexed": 0, "failed": 0}

        if namespace is None:
            namespace = config.pinecone.default_namespace

        if batch_size is None:
            batch_size = config.embedding.batch_size

        logger.info(
            f"Indexing {len(chunks)} chunks to namespace '{namespace}' "
            f"(batch_size: {batch_size})"
        )

        # Convert to Documents
        documents = [chunk.to_document() for chunk in chunks]

        # Separate parent and child chunks for better organization
        parent_docs = [
            doc for doc in documents
            if doc.metadata.get("chunk_type") == "parent"
        ]
        child_docs = [
            doc for doc in documents
            if doc.metadata.get("chunk_type") == "child"
        ]

        logger.info(f"Indexing {len(parent_docs)} parent chunks and {len(child_docs)} child chunks")

        stats = {
            "indexed": 0,
            "failed": 0,
            "parent_chunks": 0,
            "child_chunks": 0,
            "errors": []
        }

        # Index in batches
        try:
            # Index parent chunks
            if parent_docs:
                parent_stats = self._index_documents_batch(
                    parent_docs,
                    namespace,
                    batch_size,
                    chunk_type="parent"
                )
                stats["parent_chunks"] = parent_stats["indexed"]
                stats["indexed"] += parent_stats["indexed"]
                stats["failed"] += parent_stats["failed"]
                stats["errors"].extend(parent_stats["errors"])

            # Index child chunks
            if child_docs:
                child_stats = self._index_documents_batch(
                    child_docs,
                    namespace,
                    batch_size,
                    chunk_type="child"
                )
                stats["child_chunks"] = child_stats["indexed"]
                stats["indexed"] += child_stats["indexed"]
                stats["failed"] += child_stats["failed"]
                stats["errors"].extend(child_stats["errors"])

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            stats["errors"].append(str(e))

        logger.info(
            f"Indexing complete: {stats['indexed']} indexed, "
            f"{stats['failed']} failed"
        )

        return stats

    def _index_documents_batch(
        self,
        documents: List[Document],
        namespace: str,
        batch_size: int,
        chunk_type: str = "child"
    ) -> Dict[str, Any]:
        """
        Index documents in batches

        Args:
            documents: List of Document objects
            namespace: Pinecone namespace
            batch_size: Batch size
            chunk_type: Type of chunks being indexed

        Returns:
            Dictionary with batch indexing stats
        """
        stats = {
            "indexed": 0,
            "failed": 0,
            "errors": []
        }

        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Indexing {chunk_type} chunks"):
            batch = documents[i:i + batch_size]

            try:
                # Create or get vector store for this namespace
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=namespace if self.use_namespaces else None,
                    pinecone_api_key=config.pinecone.api_key
                )

                # Add documents to Pinecone
                vector_store.add_documents(batch)

                stats["indexed"] += len(batch)

                # Small delay to avoid rate limits
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Batch indexing error: {e}")
                stats["failed"] += len(batch)
                stats["errors"].append(f"Batch {i//batch_size}: {str(e)}")

        return stats

    def index_from_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index regular LangChain Documents (non-hierarchical)

        Args:
            documents: List of Document objects
            namespace: Optional namespace

        Returns:
            Indexing statistics
        """
        if namespace is None:
            namespace = config.pinecone.default_namespace

        logger.info(f"Indexing {len(documents)} documents to namespace '{namespace}'")

        try:
            vector_store = PineconeVectorStore.from_documents(
                documents,
                self.embeddings,
                index_name=self.index_name,
                namespace=namespace if self.use_namespaces else None,
                pinecone_api_key=config.pinecone.api_key
            )

            return {
                "indexed": len(documents),
                "failed": 0,
                "namespace": namespace
            }

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "indexed": 0,
                "failed": len(documents),
                "error": str(e)
            }

    def delete_namespace(self, namespace: str) -> bool:
        """
        Delete all vectors in a namespace

        Args:
            namespace: Namespace to delete

        Returns:
            Success boolean
        """
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=config.pinecone.api_key)
            index = pc.Index(self.index_name)

            index.delete(delete_all=True, namespace=namespace)

            logger.info(f"Deleted namespace: {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}: {e}")
            return False

    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index

        Args:
            namespace: Optional namespace to get stats for

        Returns:
            Dictionary with index statistics
        """
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=config.pinecone.api_key)
            index = pc.Index(self.index_name)

            stats = index.describe_index_stats()

            if namespace and self.use_namespaces:
                namespace_stats = stats.get("namespaces", {}).get(namespace, {})
                return {
                    "namespace": namespace,
                    "vector_count": namespace_stats.get("vector_count", 0)
                }
            else:
                return {
                    "total_vector_count": stats.get("total_vector_count", 0),
                    "namespaces": list(stats.get("namespaces", {}).keys())
                }

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    def create_vector_store(
        self,
        namespace: Optional[str] = None
    ) -> PineconeVectorStore:
        """
        Create a PineconeVectorStore instance for retrieval

        Args:
            namespace: Optional namespace to query

        Returns:
            PineconeVectorStore instance
        """
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace if self.use_namespaces else None,
            pinecone_api_key=config.pinecone.api_key
        )

    def test_connection(self) -> bool:
        """
        Test connection to Pinecone index

        Returns:
            Success boolean
        """
        try:
            stats = self.get_index_stats()
            if "error" in stats:
                return False

            logger.info(f"Pinecone connection successful: {stats}")
            return True

        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            return False
