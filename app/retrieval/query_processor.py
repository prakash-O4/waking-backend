"""
Query Processor for enhancing queries with Nepali legal term mapping,
expansion, and decomposition
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
from app.rag_config import config
from app.utils.loggers import logger


class QueryProcessor:
    """
    Processes and enhances user queries for better retrieval

    Features:
    - Domain detection (legal vs non-legal)
    - Nepali-English term mapping
    - Query expansion with synonyms
    - Query decomposition into sub-queries
    - Legal entity extraction (section numbers, dates, etc.)
    """

    def __init__(self):
        """Initialize the query processor"""
        self.llm = ChatOpenAI(
            temperature=0,
            model=config.generation.model,
            openai_api_key=config.openai_api_key
        )

        self.term_mappings = config.nepali_terms.legal_terms
        self.max_sub_queries = config.retrieval.max_sub_queries

        logger.info("QueryProcessor initialized")

    def process_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process and enhance a query

        Args:
            query: User's original query
            chat_history: Optional chat history for context

        Returns:
            Dictionary with processed query information:
            {
                "original": str,
                "expanded": str,
                "sub_queries": List[str],
                "is_legal_domain": bool,
                "entities": Dict,
                "metadata": Dict
            }
        """
        logger.info(f"Processing query: {query[:100]}")

        if chat_history is None:
            chat_history = []

        result = {
            "original": query,
            "expanded": query,
            "sub_queries": [],
            "is_legal_domain": True,
            "entities": {},
            "metadata": {}
        }

        # Step 1: Check if query is in legal domain
        is_legal = self._check_legal_domain(query, chat_history)
        result["is_legal_domain"] = is_legal

        if not is_legal:
            logger.info("Query detected as outside legal domain")
            return result

        # Step 2: Extract legal entities (sections, dates, etc.)
        entities = self._extract_legal_entities(query)
        result["entities"] = entities

        # Step 3: Expand query with term mappings
        expanded_query = self._expand_query(query)
        result["expanded"] = expanded_query

        # Step 4: Decompose into sub-queries
        if config.retrieval.enable_query_expansion:
            sub_queries = self._decompose_query(query, chat_history)
            result["sub_queries"] = sub_queries[:self.max_sub_queries]

        # Step 5: Add metadata
        result["metadata"] = {
            "has_entities": bool(entities),
            "num_sub_queries": len(result["sub_queries"]),
            "query_length": len(query),
            "contains_nepali": bool(re.search(r'[\u0900-\u097F]', query))
        }

        logger.info(
            f"Query processed: legal={is_legal}, "
            f"entities={len(entities)}, sub_queries={len(result['sub_queries'])}"
        )

        return result

    def _check_legal_domain(
        self,
        query: str,
        chat_history: List[Dict]
    ) -> bool:
        """
        Check if query is within legal domain

        Args:
            query: User query
            chat_history: Chat history for context

        Returns:
            True if query is legal-related
        """
        # Quick check: if query contains legal terms, likely legal
        legal_indicators = [
            "कानून", "ऐन", "संविधान", "दफा", "धारा", "अधिकार",
            "न्यायालय", "अदालत", "मुद्दा", "law", "act", "constitution",
            "rights", "court", "legal", "section", "article"
        ]

        query_lower = query.lower()
        if any(term in query_lower for term in legal_indicators):
            return True

        # Use LLM for ambiguous cases
        conversation_context = "\n".join([
            f"Human: {entry.get('question', '')}\nAI: {entry.get('answer', '')}"
            for entry in chat_history[-3:]  # Last 3 turns
        ])

        prompt = f"""Determine if this query is about Nepal's laws, legal system, or constitution.

Conversation history:
{conversation_context}

Current query: {query}

Is this query within the legal domain? Answer with just "YES" or "NO"."""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            return answer == "YES"

        except Exception as e:
            logger.warning(f"Domain detection LLM call failed: {e}")
            # Default to True if LLM fails (better to try retrieval than block)
            return True

    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from query (section numbers, dates, etc.)

        Args:
            query: User query

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract section numbers (दफा)
        dafa_matches = re.findall(r'दफा\s*([०-९\d]+)', query)
        if dafa_matches:
            entities["sections"] = dafa_matches

        # Extract article numbers (धारा)
        dhara_matches = re.findall(r'धारा\s*([०-९\d]+)', query)
        if dhara_matches:
            entities["articles"] = dhara_matches

        # Extract chapter numbers (परिच्छेद)
        chapter_matches = re.findall(r'परिच्छेद\s*([०-९\d]+)', query)
        if chapter_matches:
            entities["chapters"] = chapter_matches

        # Extract part numbers (भाग)
        part_matches = re.findall(r'भाग\s*([०-९\d]+)', query)
        if part_matches:
            entities["parts"] = part_matches

        # Extract years
        year_matches = re.findall(r'[२०]{2}[०-९]{2}', query)
        if year_matches:
            entities["years"] = year_matches

        return entities

    def _expand_query(self, query: str) -> str:
        """
        Expand query with Nepali-English term mappings

        Args:
            query: Original query

        Returns:
            Expanded query with synonyms
        """
        expanded_terms = []

        # Check each word in the query
        words = query.split()

        for word in words:
            expanded_terms.append(word)

            # Check if word has mappings
            if word in self.term_mappings:
                synonyms = self.term_mappings[word]
                # Add first 2 synonyms to avoid making query too long
                expanded_terms.extend(synonyms[:2])

        # Join back, removing duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        expanded = " ".join(unique_terms)

        # If expansion made it too long, return original
        if len(expanded) > len(query) * 2:
            return query

        return expanded

    def _decompose_query(
        self,
        query: str,
        chat_history: List[Dict]
    ) -> List[str]:
        """
        Decompose complex query into focused sub-queries

        Args:
            query: Original query
            chat_history: Chat history for context

        Returns:
            List of sub-queries (max 3)
        """
        conversation_context = "\n".join([
            f"Human: {entry.get('question', '')}\nAI: {entry.get('answer', '')}"
            for entry in chat_history[-3:]
        ])

        prompt = f"""Break down this Nepali legal query into specific sub-queries for better information retrieval.

Conversation history:
{conversation_context}

Current query: {query}

Create up to 3 focused sub-queries that:
1. Are specific and actionable
2. Cover different aspects of the main query
3. Are suitable for searching legal documents
4. Are in the same language as the original query

Return ONLY the sub-queries, one per line. If the query is already focused, return just 1-2 sub-queries."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Parse sub-queries
            sub_queries = [
                q.strip()
                for q in content.split('\n')
                if q.strip() and not q.strip().startswith('#')
            ]

            # Remove numbering if present (1., 2., etc.)
            sub_queries = [
                re.sub(r'^\d+[\.\)]\s*', '', q)
                for q in sub_queries
            ]

            # Filter out empty and very short queries
            sub_queries = [
                q for q in sub_queries
                if len(q) > 5
            ]

            return sub_queries[:self.max_sub_queries]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]  # Fallback to original query

    def get_search_queries(self, processed_query: Dict[str, Any]) -> List[str]:
        """
        Get list of all queries to search (original + expanded + sub-queries)

        Args:
            processed_query: Result from process_query()

        Returns:
            List of all search queries
        """
        if not processed_query["is_legal_domain"]:
            return []

        queries = [processed_query["original"]]

        # Add expanded if different from original
        if processed_query["expanded"] != processed_query["original"]:
            queries.append(processed_query["expanded"])

        # Add sub-queries
        queries.extend(processed_query["sub_queries"])

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries

    def detect_law_type(self, query: str) -> Optional[str]:
        """
        Detect which type of law the query is about

        Args:
            query: User query

        Returns:
            Law type namespace or None
        """
        query_lower = query.lower()

        # Check for specific law indicators
        if any(term in query_lower for term in ["संविधान", "constitution"]):
            return "constitution"

        if any(term in query_lower for term in ["अपराध", "criminal", "फौजदारी"]):
            return "criminal"

        if any(term in query_lower for term in ["देवानी", "civil"]):
            return "civil"

        if any(term in query_lower for term in ["श्रम", "labor", "labour"]):
            return "labor"

        return None  # General search across all laws
