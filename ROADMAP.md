# Wakil-G: Nepali Legal AI Chatbot — Detailed Project Documentation & Roadmap

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Component Inventory](#6-component-inventory)
7. [API Specification](#7-api-specification)
8. [Source Documents](#8-source-documents)
9. [Environment & Configuration](#9-environment--configuration)
10. [Current Gaps & Technical Debt](#10-current-gaps--technical-debt)
11. [Strategic Roadmap](#11-strategic-roadmap)
12. [Implementation Phases](#12-implementation-phases)
13. [Success Metrics & KPIs](#13-success-metrics--kpis)
14. [Risk Assessment](#14-risk-assessment)
15. [Appendix](#15-appendix)

---

## 1. Executive Summary

**Wakil-G** ("वकील-G") is a production-grade AI legal assistant specializing in Nepal's laws and constitution. Built on FastAPI with an advanced Retrieval-Augmented Generation (RAG) architecture, it enables users to ask questions in Nepali (Devanagari script) about legal documents and receive accurate, cited answers with streaming responses.

### Mission
To democratize access to Nepali legal information by providing an AI assistant that understands Nepali legal terminology, retrieves relevant legal text, and generates trustworthy answers with proper citations.

### Current Status
- **Core RAG Pipeline**: ✅ Production-ready
- **Document Ingestion**: ✅ Implemented
- **User Authentication**: ✅ Implemented (Supabase)
- **Streaming API**: ✅ Implemented
- **Testing**: ❌ Minimal coverage
- **Production Deployment**: 🔄 Partial (Docker exists, CI/CD missing)

---

## 2. Current State Assessment

### 2.1 What Works Well

| Component | Status | Quality |
|-----------|--------|---------|
| Hierarchical parent-child chunking | ✅ Active | Excellent |
| Multi-stage retrieval (vector + Cohere rerank) | ✅ Active | Excellent |
| Nepali term mapping & query expansion | ✅ Active | Good |
| Azure Document Intelligence PDF parsing | ✅ Active | Excellent |
| Pinecone namespace organization | ✅ Active | Good |
| SSE streaming responses | ✅ Active | Good |
| Supabase auth + daily quotas | ✅ Active | Good |
| Domain detection (legal vs non-legal) | ✅ Active | Good |

### 2.2 What Needs Work

| Component | Status | Priority |
|-----------|--------|----------|
| Unit/Integration tests | ❌ Missing | Critical |
| Hybrid search (BM25 + semantic) | ❌ Not implemented | High |
| Query translation (Nepali ↔ English) | ❌ Not implemented | Medium |
| Redis caching layer | 🔄 Experimental (buggy) | Medium |
| Answer evaluation metrics | ❌ Not implemented | Medium |
| User feedback loop | ❌ Not implemented | Medium |
| Production monitoring dashboards | ❌ Not implemented | Low |
| CI/CD pipeline | ❌ Not implemented | Low |

### 2.3 Technical Debt

| Issue | Severity | Description |
|-------|----------|-------------|
| Multiple competing `main.py` files | High | `main.py`, `sub_main.py`, `test_main.py` — no clear consolidation |
| Sports streaming code in codebase | Medium | `game_service.py`, `models/database.py` are unrelated leftovers |
| Empty router files | Low | `app/routers/auth.py`, `app/routers/chat.py` are empty |
| Missing root README | Low | `README.md` is empty |
| Redis vector store syntax errors | Medium | `test_main.py` has RediSearch query issues |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│         (Web App / Mobile App / curl / Any HTTP client)                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼ HTTP + SSE
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ POST /ask   │  │ GET /       │  │ Auth MW     │  │ CORS Middleware     │ │
│  │ (Streaming) │  │ (Health)    │  │ (JWT)       │  │                     │ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────┬────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL ORCHESTRATOR                                │
│  ┌─────────────────────┐              ┌─────────────────────────────────────┐ │
│  │   QUERY PROCESSOR   │              │     ADVANCED RETRIEVER              │ │
│  │  • Domain detection │─────────────▶│  • Vector search (k=20)            │ │
│  │  • Term mapping     │              │  • Cohere reranking (top 8)        │ │
│  │  • Query expansion  │              │  • Parent chunk fetch              │ │
│  │  • Decomposition    │              │  • Final selection (top 5)         │ │
│  │  • Entity extraction│              │  • Context assembly                │ │
│  └─────────────────────┘              └─────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────────┐
│      PINECONE VECTOR DB      │  │         SUPABASE (Auth + History)        │
│  Index: wakil-g              │  │  • User authentication (JWT)             │
│  Dim: 1536 (cosine)          │  │  • Chat history storage                  │
│  Namespaces:                 │  │  • Daily quota tracking (20/day)         │
│    • constitution            │  │  • User profiles                         │
│    • criminal                │  └──────────────────────────────────────────┘
│    • civil                   │
│    • labor                   │
│    • general                 │
└──────────────────────────────┘
          ▲
          │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                                      │
│                                                                              │
│  PDF ──▶ Azure Document Intelligence ──▶ Markdown ──▶ Hierarchical Chunker     │
│                                                          │                   │
│                                                          ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Metadata        │  │ Quality         │  │ Pinecone Indexer            │  │
│  │ Enricher        │◀─│ Validator       │◀─│ (with namespace)            │  │
│  │ (regex + LLM)   │  │ (score > 0.6)   │  │ (batch upsert)              │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Generation Pipeline

```
Retrieved Context + User Query + Chat History
          │
          ▼
┌─────────────────────────────────────────────────┐
│           PROMPT CONSTRUCTION                    │
│  System: Legal expert persona + guidelines       │
│  Context: Retrieved documents with citations     │
│  History: Last 10 chat turns                     │
│  Question: Current user query                    │
└─────────────────────────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────┐
│         GPT-4o-mini (streaming)                  │
│  Temperature: 0.0 (deterministic)                │
│  Max tokens: 2000                                │
│  Streaming: True (SSE)                           │
└─────────────────────────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────┐
│           TOKEN BUFFER                           │
│  Buffers partial tokens to output complete words │
│  Prevents mid-word streaming cuts                │
└─────────────────────────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────┐
│           SSE EVENTS                             │
│  event: start → data → data → ... → end         │
│  Final event includes source citations          │
└─────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

### 4.1 Core Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Runtime** | Python | 3.10+ | Backend language |
| **Framework** | FastAPI | Latest | REST API framework |
| **Server** | Uvicorn | 0.32.0 | ASGI server |
| **LLM** | OpenAI GPT-4o-mini | Latest | Answer generation |
| **Embeddings** | OpenAI text-embedding-3-small | Latest | Text vectorization (1536 dims) |
| **Vector DB** | Pinecone | ≥5.0.0 | Semantic search |
| **Reranking** | Cohere rerank-multilingual-v3.0 | ≥5.5.0 | Cross-encoder reranking |
| **PDF Parsing** | Azure Document Intelligence | 1.0.0b1 | PDF → Markdown |
| **Auth/DB** | Supabase | 2.9.1 | User auth + chat history |
| **Orchestration** | LangChain | ≥0.2.0 | LLM pipeline framework |

### 4.2 Supporting Libraries

| Library | Purpose |
|---------|---------|
| `pydantic-settings` | Configuration management |
| `pyjwt` / `python-jose` | JWT token handling |
| `python-dotenv` | Environment variables |
| `tenacity` | Retry logic with exponential backoff |
| `loguru` | Structured logging |
| `ujson` | Fast JSON parsing |
| `tqdm` | Progress bars for ingestion |

### 4.3 Infrastructure

| Component | Technology | Status |
|-----------|-----------|--------|
| **Containerization** | Docker | ✅ Implemented |
| **Cloud Registry** | AWS ECR | 🔄 Configured |
| **Monitoring** | LangSmith | ✅ Available |
| **Caching (exp)** | Redis | 🔄 Experimental |
| **CI/CD** | GitHub Actions | ❌ Not implemented |
| **Load Balancer** | AWS ALB | ❌ Not implemented |

---

## 5. Data Flow Diagrams

### 5.1 Query Flow (Runtime)

```
User Query (Nepali/English)
    │
    ▼
┌─────────────────┐
│ 1. Auth Check   │──▶ Validate JWT (Supabase)
│    (main.py)    │──▶ Check daily quota (20/day)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Query Process │──▶ Domain detection (legal vs non-legal)
│ (query_processor)│──▶ Nepali term expansion
│                 │──▶ Entity extraction (दफा, धारा, etc.)
│                 │──▶ Query decomposition (max 3 sub-queries)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Namespace     │──▶ Detect law type from query
│    Detection    │──▶ Route to: constitution / criminal / civil / labor / general
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Vector Search │──▶ Pinecone similarity search (k=20, child chunks only)
│   (Pinecone)    │──▶ Filter by namespace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Reranking    │──▶ Cohere rerank-multilingual-v3.0
│   (Cohere API)  │──▶ Score each (query, chunk) pair
│                 │──▶ Select top 8
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Parent Fetch  │──▶ Lookup parent chunks by parent_id
│                 │──▶ Deduplicate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. Final Select  │──▶ Top 5 chunks by reranker score
│                 │──▶ Assemble context (child + parent content)
│                 │──▶ Prepare source citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 8. Generation   │──▶ Format prompt with context + history + question
│   (GPT-4o-mini) │──▶ Stream tokens via SSE
│                 │──▶ Buffer tokens for complete words
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 9. Response     │──▶ event: start
│   (SSE Stream)  │──▶ event: data (word by word)
│                 │──▶ event: end (with sources array)
└─────────────────┘
```

### 5.2 Ingestion Flow (One-time/Batch)

```
PDF Source File
    │
    ▼
┌─────────────────────────┐
│ Azure Document          │──▶ prebuilt-layout model
│ Intelligence            │──▶ Nepali locale (ne)
│                         │──▶ Table extraction
│                         │──▶ Header/footer detection
└───────────┬─────────────┘
            │
            ▼ Markdown
┌─────────────────────────┐
│ Text Cleaning           │──▶ Unicode normalization (NFC)
│                         │──▶ Zero-width char removal
│                         │──▶ Devanagari conjunct fixing
│                         │──▶ Punctuation spacing (। ॥)
│                         │──▶ Whitespace normalization
│                         │──▶ Header/footer removal
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Hierarchical Chunker    │──▶ Parent chunks: ~2000 tokens
│                         │──▶ Child chunks: ~800 tokens
│                         │──▶ Overlap: 100-200 tokens
│                         │──▶ UUID relationships (parent_id, child_ids)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Metadata Enricher       │──▶ Regex: law names, parts, chapters, sections
│ (regex + LLM)           │──▶ LLM-assisted: edge case titles, dates
│                         │──▶ Cross-reference detection
│                         │──▶ Keyword extraction
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Quality Validator       │──▶ Nepali chars > 30%
│                         │──▶ Length: 50-4000 chars
│                         │──▶ Content quality (not just noise)
│                         │──▶ Legal structure markers present
│                         │──▶ Score: 0-1 (threshold: 0.6)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Pinecone Indexer        │──▶ Batch upsert (default: 100 chunks)
│                         │──▶ Separate parent/child indexing
│                         │──▶ Namespace assignment
│                         │──▶ Metadata + vector embedding
└─────────────────────────┘
```

---

## 6. Component Inventory

### 6.1 Ingestion Pipeline (`app/ingestion/`)

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| `document_processor.py` | ~200 | PDF → Markdown via Azure DI | ✅ Production |
| `hierarchical_chunker.py` | ~150 | Parent-child chunk hierarchy | ✅ Production |
| `metadata_enricher.py` | ~180 | Legal metadata extraction | ✅ Production |
| `quality_validator.py` | ~120 | Chunk quality scoring | ✅ Production |
| `pinecone_indexer.py` | ~140 | Batch indexing to Pinecone | ✅ Production |
| `__init__.py` | ~30 | Module exports | ✅ Production |

### 6.2 Retrieval Pipeline (`app/retrieval/`)

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| `query_processor.py` | ~250 | Query enhancement | ✅ Production |
| `advanced_retriever.py` | ~200 | Multi-stage retrieval | ✅ Production |
| `retrieval_orchestrator.py` | ~180 | Pipeline coordination | ✅ Production |
| `__init__.py` | ~20 | Module exports | ✅ Production |

### 6.3 API Layer (`app/`)

| File | Lines | Purpose | Quality | Notes |
|------|-------|---------|---------|-------|
| `main.py` | ~326 | **Primary production API** | ✅ Production | Uses RetrievalOrchestrator |
| `sub_main.py` | ~400 | Optimized variant with Redis | 🔄 Experimental | MMR retrieval, semantic reranking |
| `test_main.py` | ~500 | Redis vector store + planner | 🔄 Experimental | Has syntax bugs |
| `sse.py` | ~150 | Mock SSE for testing | ⚠️ Dev only | Unrelated test scenarios |
| `config.py` | ~50 | Basic Pydantic settings | ✅ Production | API key generation |
| `rag_config.py` | ~262 | Centralized RAG config | ✅ Production | All tunables in one place |

### 6.4 Utilities (`app/utils/`)

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| `helpers.py` | ~200 | SupabaseHelper (auth, quotas, history) | ✅ Production |
| `token_buffer.py` | ~80 | Token buffering for SSE | ✅ Production |
| `loggers.py` | ~50 | Loguru configuration | ✅ Production |
| `security.py` | ~40 | HMAC signature verification | ✅ Production |
| `pdf_to_markdown.py` | ~200 | Azure DI v1 converter | ✅ Production |
| `pdf_to_markdown_v2.py` | ~180 | Azure Form Recognizer (legacy) | ⚠️ Legacy |
| `ingest_data.py` | ~100 | Legacy PyPDFLoader ingestion | ⚠️ Legacy |
| `ingest_redis.py` | ~120 | Redis vector store ingestion | 🔄 Experimental |

### 6.5 Services (`app/services/`)

| File | Lines | Purpose | Quality | Notes |
|------|-------|---------|---------|-------|
| `appwrite_service.py` | ~30 | Appwrite client | ❌ Unused | Dead code |
| `game_service.py` | ~200 | Sports streaming API | ❌ Unrelated | Leftover from other project |
| `langchain_service.py` | ~0 | Empty | ❌ Empty | Dead code |

### 6.6 Models (`app/models/`)

| File | Lines | Purpose | Quality | Notes |
|------|-------|---------|---------|-------|
| `database.py` | ~100 | SQLAlchemy models | ❌ Unrelated | Game, StreamDetail (sports) |
| `schemas.py` | ~80 | Pydantic schemas | ❌ Unrelated | Game/stream schemas |
| `chat.py` | ~0 | Empty | ❌ Empty | Planned but unused |
| `user.py` | ~0 | Empty | ❌ Empty | Planned but unused |

### 6.7 Routers (`app/routers/`)

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| `auth.py` | ~0 | Empty | ❌ Not implemented |
| `chat.py` | ~0 | Empty | ❌ Not implemented |

### 6.8 Scripts (`scripts/`)

| File | Purpose | Quality |
|------|---------|---------|
| `ingest_documents.py` | Batch document ingestion | ✅ Production |
| `example_usage.py` | Configuration validation + examples | ✅ Production |
| `ingest_case_law.py` | Case law ingestion | ✅ Production |
| `delete_by_source_id.py` | Delete chunks by source | ✅ Production |

### 6.9 Tests (`tests/`)

| File | Purpose | Quality | Notes |
|------|---------|---------|-------|
| `test_auth.py` | Auth tests | ⚠️ Minimal | Mostly empty |
| `test_chat.py` | Chat tests | ⚠️ Minimal | Mostly empty |
| `test_helpers.py` | Helper tests | ⚠️ Minimal | Mostly empty |
| `test_stream.py` | Streaming tests | ⚠️ Minimal | Mock implementations |
| `streaming.py` | Streaming utilities | ⚠️ Dev only | |
| `source_stream.py` | Source streaming | ⚠️ Dev only | |
| `new_streaming.py` | New streaming | ⚠️ Dev only | |
| `config.py` | Test config | ✅ Basic | |

---

## 7. API Specification

### 7.1 Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `POST /ask` | POST | Bearer JWT | Main chat endpoint — SSE streaming |
| `GET /` | GET | None | Health check |

### 7.2 POST /ask

**Request:**
```json
{
  "question": "नेपालको संविधानमा अभिव्यक्ति स्वतन्त्रताको अधिकार के हो?"
}
```

**Headers:**
```
Authorization: Bearer <supabase_jwt_token>
Content-Type: application/json
```

**Response (SSE):**
```
event: start
data: {"start": true}

event: data
data: नेपालको

event: data
data: संविधानको

event: data
data: दफा

event: data
data: १७

event: data
data: मा
...
event: end
data: {"end": true, "sources": [{"title": "नेपालको संविधान, २०७२", "section": "दफा १७", "score": 0.892}]}
```

**Error Responses:**
| Status | Code | Description |
|--------|------|-------------|
| 404 | quota_exceeded | Daily limit of 20 queries reached |
| 429 | insufficient_quota | OpenAI API quota exhausted |
| 401 | invalid_token | JWT validation failed |
| 500 | internal_error | Unexpected server error |

### 7.3 Quota System

- **Daily Limit**: 20 queries per user per day
- **Tracking**: Stored in Supabase `config` table
- **Reset**: Daily at midnight (server timezone)

---

## 8. Source Documents

### 8.1 Current Corpus

| Document | File | Pages | Namespace | Status |
|----------|------|-------|-----------|--------|
| Nepal Constitution, 2072 | `Constitution.pdf` | ~240 | `constitution` | ✅ Indexed |
| Criminal Code (Muluki Aparadh Samhita) | `Criminal law.pdf` | ~580 | `criminal` | ✅ Indexed |
| Labor Act | `Labor law.pdf` | ~120 | `labor` | ✅ Indexed |
| Case Law Collection | `Case Law.pdf` | Unknown | `general` | ✅ Indexed |
| Public Service Broadcasting Act, 2081 | `sarbajanik.pdf` | Unknown | `general` | ✅ Indexed |

### 8.2 Document Statistics (Estimated)

| Metric | Constitution | Criminal Law | Labor Law | Total |
|--------|-------------|--------------|-----------|-------|
| Parent Chunks | ~45 | ~120 | ~30 | ~195 |
| Child Chunks | ~180 | ~480 | ~120 | ~780 |
| Total Chunks | ~225 | ~600 | ~150 | ~975 |
| Avg Quality Score | 0.92 | 0.88 | 0.85 | 0.88 |

---

## 9. Environment & Configuration

### 9.1 Required Environment Variables

```bash
# Core AI Services (Required)
OPENAI_API_KEY=sk-...                    # OpenAI API key
PINECONE_API_KEY=...                     # Pinecone API key
COHERE_API_KEY=...                       # Cohere API key for reranking

# Document Processing (Required)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://...cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...

# Authentication (Required)
SUPABASE_URL=https://...supabase.co
SUPABASE_KEY=...
SUPABASE_JWT_SECRET=...

# Monitoring (Optional)
LANGCHAIN_API_KEY=...                    # LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Wakil-G

# Caching (Optional)
REDIS_URL=redis://...                     # Redis connection (experimental)
REDIS_HOST=...
REDIS_PORT=6379
REDIS_PASSWORD=...
```

### 9.2 Configuration Tunables (`app/rag_config.py`)

| Category | Parameter | Default | Range | Description |
|----------|-----------|---------|-------|-------------|
| **Embedding** | `model` | text-embedding-3-small | Fixed | OpenAI embedding model |
| **Embedding** | `dimensions` | 1536 | Fixed | Vector dimensions |
| **Embedding** | `batch_size` | 100 | 10-500 | Embedding batch size |
| **Chunking** | `parent_chunk_size` | 2000 | 1000-4000 | Parent chunk tokens |
| **Chunking** | `child_chunk_size` | 800 | 400-1500 | Child chunk tokens |
| **Retrieval** | `initial_k` | 20 | 10-50 | Initial vector search results |
| **Retrieval** | `reranker_top_k` | 8 | 5-20 | Post-rerank selection |
| **Retrieval** | `final_top_k` | 5 | 3-10 | Final context chunks |
| **Retrieval** | `use_reranker` | True | Bool | Enable Cohere reranking |
| **Generation** | `model` | gpt-4o-mini | Fixed | LLM model |
| **Generation** | `temperature` | 0.0 | 0.0-1.0 | Creativity (0 = deterministic) |
| **Generation** | `max_tokens` | 2000 | 500-4000 | Max response length |

---

## 10. Current Gaps & Technical Debt

### 10.1 Critical Gaps (Block Production)

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 1 | **No unit tests** | Cannot verify changes, high regression risk | 2-3 days |
| 2 | **No integration tests** | Cannot verify end-to-end pipeline | 2-3 days |
| 3 | **No load testing** | Unknown capacity limits | 1-2 days |
| 4 | **Multiple main.py files** | Confusion, maintenance burden | 1 day |

### 10.2 High-Priority Gaps

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 5 | **No hybrid search (BM25)** | Weak exact-match for section numbers | 3-5 days |
| 6 | **No query translation** | Mixed-language queries perform poorly | 2-3 days |
| 7 | **No answer evaluation** | Cannot measure quality improvements | 2-3 days |
| 8 | **No user feedback loop** | Cannot learn from mistakes | 3-5 days |
| 9 | **Unrelated sports code** | Codebase bloat, confusion | 0.5 day |

### 10.3 Medium-Priority Gaps

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 10 | **Redis caching not production-ready** | Higher costs, slower responses | 3-5 days |
| 11 | **No CI/CD pipeline** | Manual deployment, error-prone | 2-3 days |
| 12 | **No production monitoring** | Blind to issues | 2-3 days |
| 13 | **Empty router files** | Missing modular API structure | 1-2 days |
| 14 | **Missing root README** | Poor onboarding | 0.5 day |

---

## 11. Strategic Roadmap

### Vision Statement
> "Wakil-G will be the most trusted and comprehensive AI legal assistant for Nepal, serving citizens, lawyers, and students with instant, accurate, and cited legal information in Nepali and English."

### Strategic Pillars

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STRATEGIC PILLARS                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   ACCURACY      │   COVERAGE      │   TRUST       │   ACCESSIBILITY         │
│                 │                 │               │                         │
│ • Hybrid search │ • More laws     │ • Citations   │ • Nepali + English      │
│ • Better rerank │ • Case law      │ • Source links│ • Voice input           │
│ • Eval metrics  │ • Regulations   │ • Confidence  │ • Mobile optimization   │
│ • Feedback loop │ • International │ scores        │ • Offline mode          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 12. Implementation Phases

### Phase 0: Foundation Cleanup (Week 1-2) — 🔴 CRITICAL

**Goal:** Stabilize codebase, remove debt, establish testing baseline.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P0.1 | Consolidate `main.py` — merge `sub_main.py` and `test_main.py` features or delete them | 1 day | Backend |
| P0.2 | Remove all sports streaming code (`game_service.py`, `models/database.py`, `models/schemas.py`) | 0.5 day | Backend |
| P0.3 | Delete empty/legacy files (`langchain_service.py`, `appwrite_service.py`, `routers/*.py` if unused) | 0.5 day | Backend |
| P0.4 | Write comprehensive root `README.md` | 0.5 day | Docs |
| P0.5 | Set up pytest framework with `conftest.py` | 0.5 day | Backend |
| P0.6 | Write unit tests for `document_processor.py` | 1 day | Backend |
| P0.7 | Write unit tests for `hierarchical_chunker.py` | 1 day | Backend |
| P0.8 | Write unit tests for `query_processor.py` | 1 day | Backend |
| P0.9 | Write unit tests for `advanced_retriever.py` | 1 day | Backend |
| P0.10 | Write integration test for full `/ask` endpoint | 1 day | Backend |
| P0.11 | Set up GitHub Actions CI (lint + test) | 0.5 day | DevOps |

**Deliverables:**
- Single, clean `main.py`
- >80% test coverage on core pipeline
- CI pipeline running on every PR
- Updated README with setup instructions

---

### Phase 1: Retrieval Enhancement (Week 3-5) — 🟠 HIGH

**Goal:** Improve retrieval accuracy and coverage.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P1.1 | Implement hybrid search: BM25 + semantic (Pinecone hybrid or Elasticsearch) | 3 days | Backend |
| P1.2 | Add query translation: Nepali ↔ English auto-detection and translation | 2 days | Backend |
| P1.3 | Implement multilingual embeddings experiment (XLM-RoBERTa / LaBSE) | 2 days | Backend |
| P1.4 | Add semantic chunking using sentence transformers for topic shift detection | 2 days | Backend |
| P1.5 | Implement query expansion with Nepali synonyms (expand current mapping) | 1 day | Backend |
| P1.6 | Add entity-based filtering (e.g., if दफा १७ mentioned, boost that section) | 1 day | Backend |
| P1.7 | Build golden Q&A evaluation set (30-50 Nepali legal questions) | 2 days | Legal + Backend |
| P1.8 | Implement automated retrieval evaluation (Recall@K, nDCG@K) | 1 day | Backend |

**Deliverables:**
- Hybrid search operational
- Query translation working
- Evaluation framework with golden set
- Retrieval precision >90% (up from ~88%)

---

### Phase 2: Trust & Citations (Week 6-7) — 🟡 MEDIUM

**Goal:** Make answers verifiable and trustworthy.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P2.1 | Enforce inline citations in prompt (every fact must cite source) | 1 day | Backend |
| P2.2 | Add confidence scores to each citation | 1 day | Backend |
| P2.3 | Implement answer faithfulness evaluation (LLM-as-judge) | 2 days | Backend |
| P2.4 | Add source snippet preview in response (show exact quoted text) | 1 day | Backend |
| P2.5 | Build user feedback mechanism (thumbs up/down + comment) | 2 days | Backend + Frontend |
| P2.6 | Store feedback in Supabase for continuous improvement | 1 day | Backend |
| P2.7 | Implement answer hallucination detection | 2 days | Backend |

**Deliverables:**
- Every answer has inline citations with confidence
- User feedback collection active
- Hallucination detection running
- Faithfulness score >95%

---

### Phase 3: Performance & Scale (Week 8-9) — 🟡 MEDIUM

**Goal:** Reduce costs and latency, handle more users.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P3.1 | Fix Redis caching layer (fix syntax bugs in `test_main.py`) | 2 days | Backend |
| P3.2 | Implement query result caching (cache frequent queries) | 1 day | Backend |
| P3.3 | Implement embedding caching | 1 day | Backend |
| P3.4 | Add smart reranker skipping (skip if vector scores are already high) | 1 day | Backend |
| P3.5 | Implement batch query processing API | 2 days | Backend |
| P3.6 | Add rate limiting per endpoint (Redis-based) | 1 day | Backend |
| P3.7 | Load testing with Locust (100 concurrent users) | 1 day | DevOps |
| P3.8 | Optimize prompt size (reduce context window waste) | 1 day | Backend |

**Deliverables:**
- Redis caching production-ready
- 30-40% cost reduction from caching
- P95 latency <2s
- Supports 100+ concurrent users

---

### Phase 4: Document Expansion (Week 10-12) — 🟡 MEDIUM

**Goal:** Expand legal document coverage.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P4.1 | Ingest Civil Code (Muluki Dewani Samhita, 2074) | 1 day | Backend |
| P4.2 | Ingest additional labor regulations and rules | 1 day | Backend |
| P4.3 | Ingest Nepal Bar Association guidelines | 1 day | Backend |
| P4.4 | Build case law ingestion pipeline (from `Case Law.pdf`) | 2 days | Backend |
| P4.5 | Add document versioning support (track law amendments) | 2 days | Backend |
| P4.6 | Implement automatic re-ingestion on law updates | 2 days | Backend |
| P4.7 | Add document metadata API (list all available laws) | 1 day | Backend |
| P4.8 | Build web scraper for Nepal Law Commission website | 3 days | Backend |

**Deliverables:**
- 10+ laws indexed
- Case law searchable
- Document versioning active
- Auto-update from official sources

---

### Phase 5: User Experience (Week 13-14) — 🟢 LOW

**Goal:** Improve conversational experience and accessibility.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P5.1 | Implement follow-up question suggestions | 2 days | Backend |
| P5.2 | Add clarification prompts for ambiguous queries | 2 days | Backend |
| P5.3 | Improve multi-turn conversation context handling | 1 day | Backend |
| P5.4 | Add conversation summarization for long chats | 1 day | Backend |
| P5.5 | Implement voice input (Nepali speech-to-text) | 3 days | Backend |
| P5.6 | Add response reading aloud (text-to-speech) | 2 days | Backend |
| P5.7 | Build chat export (PDF/JSON of conversation) | 1 day | Backend |
| P5.8 | Implement chat folders/organization | 2 days | Backend + Frontend |

**Deliverables:**
- Conversational AI with suggestions
- Voice input/output support
- Chat export functionality
- Better context handling

---

### Phase 6: Production Hardening (Week 15-16) — 🟡 MEDIUM

**Goal:** Make production-ready with monitoring and reliability.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P6.1 | Set up structured logging with correlation IDs | 1 day | Backend |
| P6.2 | Build custom monitoring dashboard (Grafana/CloudWatch) | 2 days | DevOps |
| P6.3 | Add alerting (latency >5s, error rate >1%) | 1 day | DevOps |
| P6.4 | Implement circuit breakers for external APIs | 1 day | Backend |
| P6.5 | Add graceful degradation (fallback to simpler retrieval) | 1 day | Backend |
| P6.6 | Set up automated backups (Pinecone + Supabase) | 1 day | DevOps |
| P6.7 | Implement API versioning strategy | 1 day | Backend |
| P6.8 | Write deployment runbook | 1 day | Docs |
| P6.9 | Set up staging environment | 1 day | DevOps |
| P6.10 | Production security audit | 2 days | Security |

**Deliverables:**
- Monitoring dashboards live
- Alerting configured
- 99.5% uptime target
- Security audit passed
- Staging + production environments

---

### Phase 7: Advanced Features (Week 17-20) — 🟢 LOW

**Goal:** Differentiate with advanced capabilities.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P7.1 | Implement legal document comparison (compare two laws) | 3 days | Backend |
| P7.2 | Build legal timeline visualization (law evolution) | 3 days | Backend + Frontend |
| P7.3 | Add "similar cases" recommendation | 2 days | Backend |
| P7.4 | Implement legal document drafting assistant | 5 days | Backend |
| P7.5 | Build court directory integration | 2 days | Backend |
| P7.6 | Add lawyer referral system | 2 days | Backend + Frontend |
| P7.7 | Implement multi-language support (Hindi, English fully) | 3 days | Backend |
| P7.8 | Build public API for third-party integrations | 3 days | Backend |
| P7.9 | Add legal news/updates feed | 2 days | Backend |
| P7.10 | Implement RAGAS automated evaluation pipeline | 2 days | Backend |

**Deliverables:**
- Document comparison tool
- Legal drafting assistant
- Public API available
- Multi-language support
- RAGAS evaluation running

---

### Phase 8: Scale & Ecosystem (Week 21-24) — 🟢 LOW

**Goal:** Scale to serve all of Nepal.

| Task | Description | Effort | Owner |
|------|-------------|--------|-------|
| P8.1 | Implement multi-tenancy (organizations) | 3 days | Backend |
| P8.2 | Add usage-based billing integration | 3 days | Backend |
| P8.3 | Build admin dashboard for law management | 5 days | Frontend |
| P8.4 | Implement custom document sets per organization | 2 days | Backend |
| P8.5 | Add SAML/SSO enterprise authentication | 2 days | Backend |
| P8.6 | Build mobile SDK (React Native/Flutter) | 5 days | Mobile |
| P8.7 | Partner with Nepal Law Commission for official data | Ongoing | Business |
| P8.8 | Implement offline mode (sync critical laws) | 5 days | Mobile |

**Deliverables:**
- Enterprise features
- Mobile SDK
- Official data partnership
- Offline mode

---

## 13. Success Metrics & KPIs

### 13.1 Technical Metrics

| Metric | Current | Phase 1 Target | Phase 4 Target | Phase 8 Target |
|--------|---------|----------------|----------------|----------------|
| Retrieval Precision@5 | ~88% | >90% | >92% | >95% |
| Average Latency | 1.5-2.5s | <2s | <1.5s | <1s |
| P95 Latency | ~3s | <3s | <2.5s | <2s |
| Test Coverage | ~5% | >80% | >85% | >90% |
| Uptime | Unknown | 99% | 99.5% | 99.9% |
| Error Rate | Unknown | <2% | <1% | <0.5% |

### 13.2 Quality Metrics

| Metric | Current | Phase 2 Target | Phase 4 Target | Phase 8 Target |
|--------|---------|----------------|----------------|----------------|
| Answer Faithfulness | Unknown | >90% | >93% | >95% |
| Citation Accuracy | Unknown | >85% | >90% | >95% |
| User Satisfaction | Unknown | >3.5/5 | >4/5 | >4.5/5 |
| Hallucination Rate | Unknown | <10% | <5% | <2% |

### 13.3 Business Metrics

| Metric | Current | Phase 4 Target | Phase 8 Target |
|--------|---------|----------------|----------------|
| Documents Indexed | 5 | 10+ | 50+ |
| Daily Active Users | Unknown | 500 | 10,000 |
| Queries/Day | Unknown | 5,000 | 100,000 |
| Cost/Query | ~$0.006 | ~$0.004 | ~$0.003 |

---

## 14. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OpenAI API rate limits / quota | Medium | High | Implement caching, fallback to cheaper models |
| Pinecone costs scale with data | Medium | Medium | Optimize chunk sizes, use namespaces efficiently |
| Azure DI parsing errors on new PDFs | Medium | Medium | Add fallback OCR (PyMuPDF + Tesseract) |
| Cohere API latency spikes | Medium | Medium | Add circuit breaker, skip reranker if slow |
| Nepali font encoding issues | Medium | High | Implement Preeti font detection + conversion |
| Legal accuracy concerns | High | Critical | Always include citations, add disclaimer, lawyer review |
| Data privacy (legal queries) | Medium | High | Encrypt chat history, comply with Nepal data laws |
| Competitor launches similar product | Low | Medium | Focus on Nepali language accuracy, build moat |

---

## 15. Appendix

### A. File Structure

```
wakilg-backend/
├── app/
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── hierarchical_chunker.py
│   │   ├── metadata_enricher.py
│   │   ├── quality_validator.py
│   │   └── pinecone_indexer.py
│   ├── retrieval/              # Retrieval pipeline
│   │   ├── __init__.py
│   │   ├── query_processor.py
│   │   ├── advanced_retriever.py
│   │   └── retrieval_orchestrator.py
│   ├── utils/                  # Utilities
│   │   ├── helpers.py
│   │   ├── token_buffer.py
│   │   ├── loggers.py
│   │   ├── security.py
│   │   ├── pdf_to_markdown.py
│   │   └── pdf_to_markdown_v2.py
│   ├── services/               # Services (mostly unused)
│   ├── models/                 # Models (mostly unused)
│   ├── routers/                # API routers (empty)
│   ├── main.py                 # ✅ Production API
│   ├── sub_main.py             # 🔄 Experimental
│   ├── test_main.py            # 🔄 Experimental (buggy)
│   ├── sse.py                  # ⚠️ Test/dev
│   ├── config.py               # Basic config
│   └── rag_config.py           # Centralized RAG config
├── scripts/
│   ├── ingest_documents.py
│   ├── ingest_case_law.py
│   ├── example_usage.py
│   └── delete_by_source_id.py
├── tests/                      # ⚠️ Minimal coverage
├── source/                     # Source PDFs
├── app/processed/              # Processed markdown/chunks
├── app/logs/                   # Application logs
├── features/                   # Feature specs
├── output/                     # Generated outputs
├── .env                        # Environment variables
├── requirements.txt
├── Dockerfile
├── build_push.sh
├── README.md                   # ❌ Empty
├── QUICKSTART.md
├── RAG_IMPLEMENTATION_PLAN.md
├── ADVANCED_RAG_README.md
└── PDF_TO_MARKDOWN_README.md
```

### B. Technology Comparison

| Feature | Current | Planned Alternative | Benefit |
|---------|---------|---------------------|---------|
| Embeddings | text-embedding-3-small | XLM-RoBERTa / LaBSE | Better Nepali support |
| Vector Search | Pinecone (semantic only) | Pinecone hybrid / Elasticsearch | Exact match + semantic |
| Reranking | Cohere v3 | Pinecone reranker | Lower latency, comparable quality |
| LLM | GPT-4o-mini | Claude 3.5 Haiku / Local model | Cost reduction, privacy |
| PDF Parsing | Azure DI | PyMuPDF + Tesseract fallback | No cloud dependency, free |

### C. Glossary

| Term | Meaning |
|------|---------|
| **दफा (Dafa)** | Section (of a law) |
| **धारा (Dhara)** | Article (of constitution) / Section |
| **परिच्छेद (Parichhed)** | Chapter |
| **भाग (Bhag)** | Part |
| **ऐन (Ain)** | Act / Law |
| **संहिता (Samhita)** | Code |
| **मुलुकी (Muluki)** | National / Country-wide |
| **अपराध (Aparadh)** | Crime / Offense |
| **देवानी (Dewani)** | Civil |
| **अदालत (Adalat)** | Court |
| **मुद्दा (Mudda)** | Case |
| **उजुरी (Ujuri)** | Complaint / Petition |

### D. Quick Reference Commands

```bash
# Development
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Ingestion
python scripts/ingest_documents.py --pdf source/Constitution.pdf --namespace constitution
python scripts/ingest_documents.py --source source/ --namespace general

# Testing
python scripts/example_usage.py
pytest tests/ -v

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker build -t wakil-g .
docker run -p 8000:8000 --env-file .env wakil-g
```

---

**Document Version:** 1.0  
**Last Updated:** 2024-10-31  
**Next Review:** After Phase 0 completion  
**Author:** Wakil-G Development Team
