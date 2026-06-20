#!/usr/bin/env python3
"""
Example Usage of the Advanced RAG Pipeline

This script demonstrates how to use the complete RAG system
for querying Nepali legal documents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import RetrievalOrchestrator
from app.rag_config import config
from langchain_openai import ChatOpenAI


def example_basic_retrieval():
    """Example 1: Basic retrieval"""
    print("\n" + "="*80)
    print("Example 1: Basic Retrieval")
    print("="*80)

    # Initialize orchestrator
    orchestrator = RetrievalOrchestrator()

    # Query in Nepali
    query = "नेपालको संविधानमा अभिव्यक्ति स्वतन्त्रताको अधिकार के हो?"

    # Retrieve
    result = orchestrator.retrieve(query, k=5)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(result['documents'])} documents")
    print(f"Context length: {len(result['context'])} characters")

    print("\nSources:")
    for source in result['sources']:
        print(f"  - {source['citation']} (score: {source['score']:.3f})")

    print("\nContext preview:")
    print(result['context'][:500] + "...\n")


def example_with_chat_history():
    """Example 2: Retrieval with chat history"""
    print("\n" + "="*80)
    print("Example 2: Retrieval with Chat History")
    print("="*80)

    orchestrator = RetrievalOrchestrator()

    # Simulate chat history
    chat_history = [
        {
            "question": "नेपालको संविधान कहिले जारी भएको थियो?",
            "answer": "नेपालको संविधान २०७२ साल असोज ३ गते जारी भएको थियो।"
        }
    ]

    # Follow-up query
    query = "यसमा कति भाग छन्?"

    result = orchestrator.retrieve(query, chat_history=chat_history, k=3)

    print(f"\nChat History:")
    for entry in chat_history:
        print(f"  Q: {entry['question']}")
        print(f"  A: {entry['answer'][:100]}...")

    print(f"\nCurrent Query: {query}")
    print(f"\nRetrieved {len(result['documents'])} documents")

    print("\nMetadata:")
    print(f"  - Is legal domain: {result['metadata']['is_legal_domain']}")
    print(f"  - Retrieval time: {result['metadata']['retrieval_stats']['retrieval_time_seconds']:.2f}s")
    print(f"  - Reranker used: {result['metadata']['retrieval_stats']['reranker_used']}")


def example_complete_rag():
    """Example 3: Complete RAG with answer generation"""
    print("\n" + "="*80)
    print("Example 3: Complete RAG with Answer Generation")
    print("="*80)

    # Initialize components
    orchestrator = RetrievalOrchestrator()
    llm = ChatOpenAI(
        model=config.generation.model,
        temperature=config.generation.temperature,
        openai_api_key=config.openai_api_key
    )

    # Query
    query = "नागरिकको मौलिक अधिकारहरू के के हुन्?"

    # Retrieve context
    result = orchestrator.retrieve(query, k=5)

    if not result['documents']:
        print("No documents found!")
        return

    # Create prompt
    prompt = f"""तपाईं नेपालको कानून र संविधान सम्बन्धी विशेषज्ञ हुनुहुन्छ।

सन्दर्भ:
{result['context']}

प्रश्न: {query}

कृपया प्रश्नको विस्तृत जवाफ दिनुहोस्। जवाफमा सान्दर्भिक दफा र धारा उल्लेख गर्नुहोस्।"""

    # Generate answer
    print(f"\nQuery: {query}")
    print(f"\nGenerating answer using {len(result['documents'])} documents...")

    response = llm.invoke(prompt)
    answer = response.content

    print(f"\nAnswer:\n{answer}")

    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['citation']}")


def example_domain_detection():
    """Example 4: Domain detection (legal vs non-legal)"""
    print("\n" + "="*80)
    print("Example 4: Domain Detection")
    print("="*80)

    orchestrator = RetrievalOrchestrator()

    # Legal query
    legal_query = "दफा १७ मा के लेखिएको छ?"
    legal_result = orchestrator.retrieve(legal_query, k=3)

    print(f"\nLegal Query: {legal_query}")
    print(f"  - Is legal domain: {legal_result['metadata']['is_legal_domain']}")
    print(f"  - Documents found: {len(legal_result['documents'])}")

    # Non-legal query
    non_legal_query = "मोमो कसरी बनाउने?"
    non_legal_result = orchestrator.retrieve(non_legal_query, k=3)

    print(f"\nNon-Legal Query: {non_legal_query}")
    print(f"  - Is legal domain: {non_legal_result['metadata']['is_legal_domain']}")
    print(f"  - Documents found: {len(non_legal_result['documents'])}")


def example_retrieval_stats():
    """Example 5: Detailed retrieval statistics"""
    print("\n" + "="*80)
    print("Example 5: Retrieval Statistics")
    print("="*80)

    orchestrator = RetrievalOrchestrator()

    query = "श्रम ऐन २०७४ को मुख्य प्रावधानहरू के हुन्?"

    result = orchestrator.retrieve(query, k=5)

    print(f"\nQuery: {query}")

    stats = result['metadata']['retrieval_stats']

    print(f"\nRetrieval Statistics:")
    print(f"  - Queries executed: {stats['queries_executed']}")
    print(f"  - Initial docs retrieved: {stats['initial_docs_retrieved']}")
    print(f"  - Final docs: {stats['final_doc_count']}")
    print(f"  - Parent chunks: {stats['parent_doc_count']}")
    print(f"  - Avg score: {stats['avg_score']:.3f}")
    print(f"  - Min score: {stats['min_score']:.3f}")
    print(f"  - Max score: {stats['max_score']:.3f}")
    print(f"  - Retrieval time: {stats['retrieval_time_seconds']:.2f}s")
    print(f"  - Reranker used: {stats['reranker_used']}")
    print(f"  - Namespace: {stats['namespace']}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("Advanced RAG Pipeline - Usage Examples")
    print("="*80)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please check your .env file and ensure all required API keys are set:")
        print("  - OPENAI_API_KEY")
        print("  - PINECONE_API_KEY")
        print("  - COHERE_API_KEY (for reranking)")
        print("  - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        print("  - AZURE_DOCUMENT_INTELLIGENCE_KEY")
        sys.exit(1)

    print("\n✓ Configuration validated successfully\n")

    # Run examples
    try:
        example_basic_retrieval()
        example_with_chat_history()
        example_domain_detection()
        example_retrieval_stats()
        example_complete_rag()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
