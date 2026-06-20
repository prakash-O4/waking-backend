"""
Centralized configuration for Wakilg Advanced RAG Pipeline
"""
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    model: str = "text-embedding-3-small"  # OpenAI embedding model
    dimensions: int = 1536
    batch_size: int = 100


@dataclass
class ChunkingConfig:
    """Configuration for hierarchical chunking"""
    # Parent chunk settings
    parent_chunk_size: int = 2000  # Tokens for parent chunks (full sections)
    parent_chunk_overlap: int = 200

    # Child chunk settings
    child_chunk_size: int = 800  # Tokens for child chunks (sub-sections)
    child_chunk_overlap: int = 100

    # Separators for Nepali legal documents
    separators: List[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n## ",      # Chapter markers
                "\n\n### ",     # Section markers (Dafa)
                "\n\n",         # Paragraph breaks
                "\n",           # Line breaks
                "। ",           # Nepali sentence ending (purna biram)
                "॥ ",           # Nepali double danda
                ". ",           # English period
                " ",            # Space
                ""              # Character level
            ]


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline"""
    # Initial retrieval
    initial_k: int = 20  # Number of chunks to retrieve initially

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "cohere"  # Options: "cohere", "cross-encoder"
    reranker_top_k: int = 8  # After reranking

    # Final context
    final_top_k: int = 5  # Final chunks to use for generation
    include_parent_chunks: bool = True

    # Query processing
    max_sub_queries: int = 3
    enable_query_expansion: bool = True
    enable_domain_detection: bool = True

    # Search type
    search_type: str = "similarity"  # Options: "similarity", "mmr"

    # Namespaces (for filtering by law type)
    namespaces: Dict[str, str] = None

    def __post_init__(self):
        if self.namespaces is None:
            self.namespaces = {
                "constitution": "नेपालको संविधान",
                "criminal": "मुलुकी अपराध संहिता",
                "civil": "मुलुकी देवानी संहिता",
                "labor": "श्रम ऐन",
                "general": "सार्वजनिक"
            }


@dataclass
class GenerationConfig:
    """Configuration for response generation"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000
    streaming: bool = True

    # Citation settings
    include_citations: bool = True
    citation_format: str = "metadata"  # Options: "metadata", "inline"
    confidence_threshold: float = 0.5  # Minimum reranker score for citations


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and monitoring"""
    enable_metrics: bool = True
    enable_langsmith: bool = True

    # Retrieval metrics
    track_retrieval_latency: bool = True
    track_reranker_scores: bool = True

    # Answer quality
    enable_llm_evaluation: bool = True
    evaluation_model: str = "gpt-4o-mini"

    # Monitoring
    log_level: str = "INFO"
    metrics_export_interval: int = 3600  # seconds (1 hour)


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector store"""
    index_name: str = "wakil-g"
    api_key: Optional[str] = None
    environment: str = "gcp-starter"

    # Namespace support
    use_namespaces: bool = True
    default_namespace: str = "general"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("PINECONE_API_KEY")


@dataclass
class IngestionConfig:
    """Configuration for document ingestion"""
    # Azure Document Intelligence
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None

    # Quality validation
    min_chunk_length: int = 50  # Minimum characters per chunk
    max_chunk_length: int = 4000  # Maximum characters per chunk
    min_quality_score: float = 0.6  # Minimum quality score (0-1)

    # Processing
    batch_size: int = 10
    enable_validation: bool = True

    # Paths
    source_dir: str = "./source"
    output_dir: str = "./app/processed"
    markdown_dir: str = "./app/processed/markdown"
    chunks_dir: str = "./app/processed/chunks"

    def __post_init__(self):
        if self.azure_endpoint is None:
            self.azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        if self.azure_key is None:
            self.azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")


@dataclass
class NepaliTermMapping:
    """Nepali legal term mappings for query enhancement"""
    legal_terms: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.legal_terms is None:
            self.legal_terms = {
                # Law/Act terms
                "कानून": ["ऐन", "विधि", "law", "act"],
                "ऐन": ["कानून", "विधि", "act"],
                "विधि": ["कानून", "ऐन", "law"],

                # Constitution terms
                "संविधान": ["constitution", "राज्यको आधारभूत कानून"],

                # Rights terms
                "अधिकार": ["rights", "हक"],
                "हक": ["अधिकार", "rights"],

                # Legal sections
                "दफा": ["धारा", "section", "article"],
                "धारा": ["दफा", "section", "article"],
                "परिच्छेद": ["chapter", "अध्याय"],
                "भाग": ["part", "खण्ड"],

                # Court/Justice terms
                "अदालत": ["court", "न्यायालय"],
                "न्यायालय": ["अदालत", "court"],
                "न्याय": ["justice", "इन्साफ"],

                # Common legal actions
                "मुद्दा": ["case", "केस"],
                "उजुरी": ["complaint", "petition"],
                "अपराध": ["crime", "offense"],
            }


class RAGConfig:
    """Main configuration class that combines all configs"""

    def __init__(self):
        # Load all configurations
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.generation = GenerationConfig()
        self.evaluation = EvaluationConfig()
        self.pinecone = PineconeConfig()
        self.ingestion = IngestionConfig()
        self.nepali_terms = NepaliTermMapping()

        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

        # Feature flags
        self.features = {
            "use_reranker": True,
            "use_parent_chunks": True,
            "enable_query_expansion": True,
            "enable_caching": False,  # Redis caching (optional)
            "enable_metrics": True,
        }

    def get_feature(self, feature_name: str) -> bool:
        """Get feature flag value"""
        return self.features.get(feature_name, False)

    def set_feature(self, feature_name: str, value: bool):
        """Set feature flag value"""
        self.features[feature_name] = value

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            self.openai_api_key,
            self.pinecone.api_key,
        ]

        if not all(required_keys):
            raise ValueError("Missing required API keys in environment variables")

        if self.retrieval.use_reranker and self.retrieval.reranker_model == "cohere":
            if not self.cohere_api_key:
                raise ValueError("COHERE_API_KEY required when using Cohere reranker")

        return True


# Global configuration instance
config = RAGConfig()

# Validate on import
try:
    config.validate()
except ValueError as e:
    print(f"Warning: Configuration validation failed: {e}")
