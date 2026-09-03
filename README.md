# Wakil-G (वकील-G) 🇳🇵

> **An AI-powered legal assistant for Nepal's laws and constitution.**

Wakil-G is a RAG (Retrieval-Augmented Generation) system built with FastAPI that enables users to ask questions about Nepali legal documents in **Nepali (Devanagari script)** and receive cited JSON answers.

---

## ✨ Features

- **🇳🇵 Nepali Language Support** — Full Devanagari script handling with legal terminology understanding
- **📚 Multi-Law Coverage** — Constitution, Criminal Law, Civil Law, Labor Law, and more
- **🔍 Advanced RAG Pipeline** — Hierarchical parent-child chunking + multi-stage retrieval + Cohere reranking
- **📖 Cited Answers** — Every answer references specific legal provisions (दफा, धारा, परिच्छेद)
- **⚡ JSON Responses** — `/ask` returns one validated JSON response per request
- **🔐 User Authentication** — Supabase JWT auth with daily quota management (20 queries/day)
- **💬 Chat History** — Contextual multi-turn conversations

---

## 🏗️ Architecture

```
User Query (Nepali)
    │
    ▼
┌─────────────────┐     ┌──────────────────────┐
│  FastAPI /ask   │────▶│  Query Processor     │
│  (JSON response)│     │  • Domain detection  │
└─────────────────┘     │  • Term expansion    │
                        │  • Query decomposition│
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  Advanced Retriever  │
                        │  • Vector search (k=20)
                        │  • Cohere rerank     │
                        │  • Parent fetch      │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  GPT-4o-mini         │
                        │  (JSON claims)       │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  JSON Response       │
                        │  + Source Citations  │
                        └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys: OpenAI, Pinecone, Cohere, Azure Document Intelligence, Supabase

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Required
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
COHERE_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://...cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
SUPABASE_URL=https://...supabase.co
SUPABASE_KEY=...
SUPABASE_JWT_SECRET=...

# Optional
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

### 3. Ingest Documents

```bash
# Ingest a single PDF
python scripts/ingest_documents.py --pdf source/Constitution.pdf --namespace constitution

# Or ingest all PDFs
python scripts/ingest_documents.py --source source/ --namespace general
```

### 4. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test the Endpoint

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SUPABASE_JWT" \
  -d '{"question": "नेपालको संविधानमा मौलिक अधिकारहरू के के हुन्?"}'
```

---

## 📁 Project Structure

```
wakilg-backend/
├── app/
│   ├── main.py                    # Production FastAPI app
│   ├── config.py                  # Settings management
│   ├── rag_config.py              # Centralized RAG configuration
│   ├── ingestion/                 # Document ingestion pipeline
│   │   ├── document_processor.py  # PDF → Markdown (Azure DI)
│   │   ├── hierarchical_chunker.py  # Parent-child chunks
│   │   ├── metadata_enricher.py   # Legal metadata extraction
│   │   ├── quality_validator.py   # Chunk quality scoring
│   │   └── pinecone_indexer.py    # Vector DB indexing
│   ├── retrieval/                 # Retrieval pipeline
│   │   ├── query_processor.py     # Query enhancement
│   │   ├── advanced_retriever.py  # Multi-stage retrieval
│   │   └── retrieval_orchestrator.py  # Pipeline coordinator
│   └── utils/                     # Utilities
│       ├── helpers.py             # Supabase auth, quotas, history
│       ├── token_buffer.py        # unused; reserved for future streaming
│       ├── loggers.py             # Logging configuration
│       ├── security.py            # HMAC verification
│       └── pdf_to_markdown.py     # Azure DI PDF converter
├── scripts/
│   ├── ingest_documents.py        # Batch ingestion script
│   ├── ingest_case_law.py        # Case law ingestion
│   ├── example_usage.py           # Configuration validation
│   └── delete_by_source_id.py     # Chunk deletion utility
├── source/                        # Source PDFs
│   ├── Constitution.pdf
│   ├── Criminal law.pdf
│   ├── Labor law.pdf
│   ├── Case Law.pdf
│   └── sarbajanik.pdf
├── tests/                         # Test suite (WIP)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | 10-minute setup guide |
| [ADVANCED_RAG_README.md](ADVANCED_RAG_README.md) | Full RAG pipeline documentation |
| [RAG_IMPLEMENTATION_PLAN.md](RAG_IMPLEMENTATION_PLAN.md) | Implementation phases & benchmarks |
| [PDF_TO_MARKDOWN_README.md](PDF_TO_MARKDOWN_README.md) | PDF conversion guide |
| [ROADMAP.md](ROADMAP.md) | Strategic roadmap & future phases |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI text-embedding-3-small (1536 dims) |
| **Vector DB** | Pinecone (index: `wakil-g`) |
| **Reranking** | Cohere rerank-multilingual-v3.0 |
| **PDF Parsing** | Azure Document Intelligence |
| **Auth/History** | Supabase |
| **Monitoring** | LangSmith |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Retrieval Precision@5 | ~88% |
| Average Latency | 1.5-2.5s |
| Chunk Quality Score | 0.85 avg |
| Cost per Query | ~$0.006 |

---

## 🤝 Contributing

This project is actively being developed. See [ROADMAP.md](ROADMAP.md) for planned features and phases.

---

## 📜 License

MIT License — Wakil-G Development Team

---

> **Disclaimer:** Wakil-G provides legal information for educational purposes only. It does not constitute legal advice. Always consult a qualified legal professional for specific legal matters.
