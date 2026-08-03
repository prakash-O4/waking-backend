"""
Advanced Retriever with multi-stage retrieval and Cohere reranking
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import cohere
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from app.rag_config import config
from app.utils.loggers import logger


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    documents: List[Document]
    scores: List[float]
    parent_documents: List[Document]
    metadata: Dict[str, Any]


class AdvancedRetriever:
    """
    Multi-stage retriever with:
    1. Initial vector search (retrieves child chunks)
    2. Cohere reranking for better relevance
    3. Parent chunk fetching for context
    """

    def __init__(self, namespace: Optional[str] = None):
        """
        Initialize the advanced retriever

        Args:
            namespace: Optional Pinecone namespace to search in
        """
        self.namespace = namespace or config.pinecone.default_namespace
        self.use_reranker = config.retrieval.use_reranker

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_key=config.openai_api_key
        )

        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=config.pinecone.index_name,
            embedding=self.embeddings,
            namespace=self.namespace if config.pinecone.use_namespaces else None,
            pinecone_api_key=config.pinecone.api_key
        )

        # Initialize Cohere reranker if enabled
        if self.use_reranker and config.retrieval.reranker_model == "cohere":
            if not config.cohere_api_key:
                logger.warning("Cohere API key not found, disabling reranker")
                self.use_reranker = False
            else:
                self.cohere_client = cohere.Client(config.cohere_api_key)
                logger.info("Cohere reranker initialized")

        logger.info(f"AdvancedRetriever initialized (namespace={self.namespace}, reranker={self.use_reranker})")

    def retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None,
        include_parents: Optional[bool] = None
    ) -> RetrievalResult:
        """
        Retrieve documents using multi-stage pipeline

        Args:
            queries: List of search queries (original + expanded + sub-queries)
            k: Number of final documents to return
            include_parents: Whether to fetch parent chunks

        Returns:
            RetrievalResult with documents, scores, and metadata
        """
        if not queries:
            logger.warning("No queries provided for retrieval")
            return RetrievalResult(
                documents=[],
                scores=[],
                parent_documents=[],
                metadata={"error": "No queries provided"}
            )

        if k is None:
            k = config.retrieval.final_top_k

        if include_parents is None:
            include_parents = config.retrieval.include_parent_chunks

        start_time = time.time()

        logger.info(f"Retrieving with {len(queries)} queries, k={k}")

        # Stage 1: Initial vector search
        initial_docs, initial_metadata = self._initial_retrieval(queries)

        # Stage 2: Reranking (if enabled)
        if self.use_reranker and initial_docs:
            # Use the first query as the primary query for reranking
            primary_query = queries[0]
            reranked_docs, scores = self._rerank_documents(
                primary_query,
                initial_docs,
                top_k=config.retrieval.reranker_top_k
            )
        else:
            reranked_docs = initial_docs[:config.retrieval.reranker_top_k]
            scores = [1.0] * len(reranked_docs)  # Placeholder scores

        # Stage 3: Select final documents
        final_docs = reranked_docs[:k]
        final_scores = scores[:k]

        # Stage 4: Fetch parent chunks if requested
        parent_docs = []
        if include_parents and final_docs:
            parent_docs = self._fetch_parent_chunks(final_docs)

        retrieval_time = time.time() - start_time

        # Prepare metadata
        result_metadata = {
            **initial_metadata,
            "retrieval_time_seconds": retrieval_time,
            "reranker_used": self.use_reranker,
            "final_doc_count": len(final_docs),
            "parent_doc_count": len(parent_docs),
            "namespace": self.namespace
        }

        logger.info(
            f"Retrieval complete: {len(final_docs)} docs retrieved "
            f"in {retrieval_time:.2f}s (reranked={self.use_reranker})"
        )

        return RetrievalResult(
            documents=final_docs,
            scores=final_scores,
            parent_documents=parent_docs,
            metadata=result_metadata
        )

    def _initial_retrieval(
        self,
        queries: List[str]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Stage 1: Initial vector search across multiple queries

        Args:
            queries: List of search queries

        Returns:
            Tuple of (documents, metadata)
        """
        all_docs = []
        unique_doc_ids = set()

        initial_k = config.retrieval.initial_k

        for query_idx, query in enumerate(queries):
            try:
                # Use similarity_search_with_score for better control
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=initial_k,
                    filter={"chunk_type": "child"}  # Retrieve only child chunks
                )

                # Add unique documents
                for doc, score in docs_with_scores:
                    doc_id = doc.metadata.get("chunk_id", id(doc))

                    if doc_id not in unique_doc_ids:
                        # Add retrieval metadata
                        doc.metadata["retrieval_score"] = float(score)
                        doc.metadata["query_index"] = query_idx

                        all_docs.append(doc)
                        unique_doc_ids.add(doc_id)

            except Exception as e:
                logger.error(f"Vector search failed for query '{query[:50]}': {e}")
                continue

        # Sort by retrieval score (lower is better for distance metrics)
        all_docs.sort(key=lambda d: d.metadata.get("retrieval_score", float("inf")))

        metadata = {
            "queries_executed": len(queries),
            "initial_docs_retrieved": len(all_docs),
            "unique_docs": len(unique_doc_ids)
        }

        logger.info(f"Initial retrieval: {len(all_docs)} unique documents from {len(queries)} queries")

        return all_docs, metadata

    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> Tuple[List[Document], List[float]]:
        """
        Stage 2: Rerank documents using Cohere

        Args:
            query: Primary query for reranking
            documents: Documents to rerank
            top_k: Number of top documents to return after reranking

        Returns:
            Tuple of (reranked_documents, scores)
        """
        if not documents:
            return [], []

        try:
            # Prepare documents for Cohere
            doc_texts = [doc.page_content for doc in documents]

            # Call Cohere rerank API
            response = self.cohere_client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_k,
                model="rerank-multilingual-v3.0"  # Supports Nepali
            )

            # Extract reranked documents and scores
            reranked_docs = []
            scores = []

            for result in response.results:
                original_index = result.index
                score = result.relevance_score

                doc = documents[original_index]
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_index"] = result.index

                reranked_docs.append(doc)
                scores.append(float(score))

            logger.info(
                f"Reranked {len(documents)} docs -> {len(reranked_docs)} docs "
                f"(avg score: {sum(scores)/len(scores):.3f})"
            )

            return reranked_docs, scores

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original documents with placeholder scores
            return documents[:top_k], [1.0] * min(top_k, len(documents))

    def _fetch_parent_chunks(
        self,
        child_documents: List[Document]
    ) -> List[Document]:
        """
        Stage 3: Fetch parent chunks for the retrieved child chunks

        Args:
            child_documents: List of child document chunks

        Returns:
            List of parent documents
        """
        parent_docs = []
        parent_ids_seen = set()

        for child_doc in child_documents:
            parent_id = child_doc.metadata.get("parent_id")

            if not parent_id or parent_id in parent_ids_seen:
                continue

            try:
                # Search for parent chunk by ID
                results = self.vector_store.similarity_search(
                    child_doc.page_content,  # Use child content as query
                    k=20,  # Search more to ensure we find the parent
                    filter={"chunk_id": parent_id, "chunk_type": "parent"}
                )

                if results:
                    parent_doc = results[0]
                    parent_doc.metadata["is_parent_context"] = True
                    parent_docs.append(parent_doc)
                    parent_ids_seen.add(parent_id)

            except Exception as e:
                logger.warning(f"Failed to fetch parent chunk {parent_id}: {e}")
                continue

        logger.info(f"Fetched {len(parent_docs)} parent chunks")

        return parent_docs

    def set_namespace(self, namespace: str):
        """Change the search namespace"""
        self.namespace = namespace

        # Recreate vector store with new namespace
        self.vector_store = PineconeVectorStore(
            index_name=config.pinecone.index_name,
            embedding=self.embeddings,
            namespace=namespace if config.pinecone.use_namespaces else None,
            pinecone_api_key=config.pinecone.api_key
        )

        logger.info(f"Namespace changed to: {namespace}")

    def get_retrieval_stats(self, result: RetrievalResult) -> Dict[str, Any]:
        """
        Get detailed statistics about a retrieval result

        Args:
            result: RetrievalResult object

        Returns:
            Dictionary with retrieval statistics
        """
        if not result.documents:
            return {"error": "No documents in result"}

        scores = result.scores
        docs = result.documents

        return {
            "document_count": len(docs),
            "parent_count": len(result.parent_documents),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_doc_length": sum(len(d.page_content) for d in docs) / len(docs),
            "unique_sources": len(set(d.metadata.get("source_file", "") for d in docs)),
            "unique_sections": len(set(d.metadata.get("section", "") for d in docs)),
            **result.metadata
        }
