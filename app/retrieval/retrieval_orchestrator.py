"""
Retrieval Orchestrator that coordinates query processing and retrieval
"""

from typing import List, Dict, Any, Optional

from app.retrieval.query_processor import QueryProcessor
from app.retrieval.advanced_retriever import AdvancedRetriever
from app.rag_config import config
from app.utils.loggers import logger


class RetrievalOrchestrator:
    """
    Orchestrates the complete retrieval pipeline:
    1. Query processing and enhancement
    2. Multi-stage retrieval with reranking
    3. Context assembly with parent chunks
    """

    def __init__(self):
        """Initialize the retrieval orchestrator"""
        self.query_processor = QueryProcessor()
        self.retriever = AdvancedRetriever()

        logger.info("RetrievalOrchestrator initialized")

    def retrieve(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute the complete retrieval pipeline

        Args:
            query: User's query
            chat_history: Optional chat history
            k: Number of documents to retrieve

        Returns:
            Dictionary with retrieval results and metadata
        """
        if chat_history is None:
            chat_history = []

        logger.info(f"Orchestrating retrieval for query: {query[:100]}")

        # Step 1: Process query
        processed_query = self.query_processor.process_query(query, chat_history)

        # Check if query is outside legal domain
        if not processed_query["is_legal_domain"]:
            logger.info("Query outside legal domain, returning empty results")
            return {
                "documents": [],
                "context": "",
                "sources": [],
                "metadata": {
                    "is_legal_domain": False,
                    "original_query": query
                }
            }

        # Step 2: Get search queries
        search_queries = self.query_processor.get_search_queries(processed_query)

        # Step 3: Detect law type and set namespace if needed
        law_type = self.query_processor.detect_law_type(query)
        if law_type and config.pinecone.use_namespaces:
            self.retriever.set_namespace(law_type)
            logger.info(f"Using namespace: {law_type}")

        # Step 4: Retrieve documents
        retrieval_result = self.retriever.retrieve(
            queries=search_queries,
            k=k,
            include_parents=True
        )

        # Step 5: Assemble context
        context = self._assemble_context(
            retrieval_result.documents,
            retrieval_result.parent_documents
        )

        # Step 6: Prepare sources for citation
        sources = self._prepare_sources(
            retrieval_result.documents,
            retrieval_result.scores
        )

        # Step 7: Compile final result
        result = {
            "documents": retrieval_result.documents,
            "context": context,
            "sources": sources,
            "metadata": {
                "is_legal_domain": True,
                "original_query": query,
                "processed_query": processed_query,
                "retrieval_stats": self.retriever.get_retrieval_stats(retrieval_result),
                "num_documents": len(retrieval_result.documents),
                "num_parent_chunks": len(retrieval_result.parent_documents),
            }
        }

        logger.info(
            f"Retrieval complete: {len(retrieval_result.documents)} docs, "
            f"{len(context)} chars context"
        )

        return result

    def _assemble_context(
        self,
        child_docs: List,
        parent_docs: List
    ) -> str:
        """
        Assemble context from child and parent documents

        Args:
            child_docs: Retrieved child chunks
            parent_docs: Parent chunks for context

        Returns:
            Assembled context string
        """
        if not child_docs:
            return "No relevant information found."

        context_parts = []

        # Add each child document with its parent context if available
        parent_dict = {
            doc.metadata.get("chunk_id"): doc
            for doc in parent_docs
        }

        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")

            # If parent exists, use parent content for richer context
            # Otherwise, use child content
            if parent_id and parent_id in parent_dict:
                parent_doc = parent_dict[parent_id]
                context_parts.append(parent_doc.page_content)
            else:
                context_parts.append(child_doc.page_content)

        # Join with double newline
        context = "\n\n".join(context_parts)

        return context

    def _prepare_sources(
        self,
        documents: List,
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Prepare source citations from documents

        Args:
            documents: Retrieved documents
            scores: Relevance scores

        Returns:
            List of source dictionaries
        """
        sources = []

        for idx, (doc, score) in enumerate(zip(documents, scores)):
            metadata = doc.metadata

            # Create citation text
            from app.ingestion.metadata_enricher import MetadataEnricher
            enricher = MetadataEnricher(use_llm=False)
            citation = enricher.create_citation_text(metadata)

            source = {
                "index": idx,
                "citation": citation,
                "score": score,
                "metadata": {
                    "source_file": metadata.get("source_file", "Unknown"),
                    "law_name": metadata.get("law_name", ""),
                    "section": metadata.get("section", ""),
                    "chapter": metadata.get("chapter", ""),
                    "part": metadata.get("part", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                }
            }

            # Only include if score is above threshold
            if score >= config.generation.confidence_threshold:
                sources.append(source)

        return sources
