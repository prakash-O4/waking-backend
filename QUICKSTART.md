# Quick Start Guide - Advanced RAG Pipeline

This guide will get you up and running with the new advanced RAG pipeline in 10 minutes.

## Prerequisites

Ensure you have the following API keys:
- ✅ OpenAI API key
- ✅ Pinecone API key
- ✅ Cohere API key (for reranking)
- ✅ Azure Document Intelligence endpoint & key

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment (1 minute)

Add to your `.env` file:

```env
# Required for the new pipeline
COHERE_API_KEY=your_cohere_api_key_here

# Make sure these are also set
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_azure_endpoint
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_azure_key
```

## Step 3: Ingest Your First Document (3 minutes)

```bash
# Ingest a single PDF
python scripts/ingest_documents.py \
  --pdf source/Constitution.pdf \
  --namespace constitution

# Or ingest all PDFs in source/
python scripts/ingest_documents.py \
  --source source/ \
  --namespace general
```

**What happens:**
1. ✅ PDF converted to clean Markdown (Azure AI)
2. ✅ Creates parent-child hierarchical chunks
3. ✅ Enriches metadata (law names, sections, etc.)
4. ✅ Validates quality
5. ✅ Indexes to Pinecone with namespace

## Step 4: Test Retrieval (2 minutes)

```bash
# Run example queries
python scripts/example_usage.py
```

You should see output like:

```
Example 1: Basic Retrieval
==================================================
Query: नेपालको संविधानमा अभिव्यक्ति स्वतन्त्रताको अधिकार के हो?

Retrieved 5 documents
Context length: 3421 characters

Sources:
  - नेपालको संविधान, दफा १७ (score: 0.892)
  - नेपालको संविधान, दफा १८ (score: 0.845)
  ...
```

## Step 5: Start the API (1 minute)

```bash
uvicorn app.main:app --reload
```

Test the endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"question": "नागरिकको मौलिक अधिकारहरू के के हुन्?"}'
```

## Step 6: Verify Everything Works (1 minute)

Check that the new pipeline is active:

1. **Logs should show**: "AdvancedRetriever initialized"
2. **Logs should show**: "Cohere reranker initialized"
3. **API response** should include proper citations with reranker scores

## What Changed?

### Before (Old Pipeline)
```
Query → Simple decomposition → Vector search (k=3) →
Length-based ranking → Response
```

### After (New Pipeline)
```
Query → Enhanced processing (term mapping + decomposition) →
Vector search (k=20) → Cohere reranking (top 8) →
Parent chunk fetching → Context assembly → Response
```

## Key Features Now Available

### 1. Hierarchical Chunks
- Parent chunks: Full sections for context (~2000 tokens)
- Child chunks: Precise sub-sections for retrieval (~800 tokens)
- Best of both worlds: precision + context

### 2. Cohere Reranking
- Uses multilingual reranker (supports Nepali)
- Improves relevance scores by ~30-40%
- Automatically filters low-quality results

### 3. Namespace Organization
- Organize docs by law type: `constitution`, `criminal`, `civil`, `labor`
- Faster retrieval with targeted search
- Better organization

### 4. Quality Validation
- Automatic validation before indexing
- Rejects low-quality chunks
- Ensures clean, useful content

### 5. Enhanced Metadata
- Legal structure extraction (Part, Chapter, Dafa)
- Cross-reference detection
- Better citations in responses

## Configuration

### Adjust Retrieval Settings

Edit `app/rag_config.py`:

```python
# Get more results
config.retrieval.final_top_k = 8  # Default: 5

# Disable reranker (faster but less accurate)
config.retrieval.use_reranker = False  # Default: True

# Adjust chunk sizes
config.chunking.child_chunk_size = 1000  # Default: 800
config.chunking.parent_chunk_size = 2500  # Default: 2000
```

### Feature Flags

```python
from app.rag_config import config

# Enable/disable features
config.set_feature("use_reranker", True)
config.set_feature("enable_query_expansion", True)
config.set_feature("enable_metrics", True)
```

## Common Commands

### Ingestion

```bash
# Single document with namespace
python scripts/ingest_documents.py --pdf source/doc.pdf --namespace criminal

# Batch ingestion
python scripts/ingest_documents.py --source source/ --namespace general

# Skip validation (faster, not recommended)
python scripts/ingest_documents.py --pdf doc.pdf --skip-validation

# Save stats
python scripts/ingest_documents.py --pdf doc.pdf --output-stats stats.json
```

### Testing

```bash
# Run all examples
python scripts/example_usage.py

# Test specific features
python -c "from app.retrieval import RetrievalOrchestrator; \
           o = RetrievalOrchestrator(); \
           print(o.retrieve('test query'))"
```

### Monitoring

```bash
# Check API logs
tail -f app/logs/api.log

# Check Pinecone index stats
python -c "from app.ingestion import PineconeIndexer; \
           i = PineconeIndexer(); \
           print(i.get_index_stats())"
```

## Troubleshooting

### Issue: "Cohere API key not found"

**Solution**: Add `COHERE_API_KEY` to your `.env` file

### Issue: "No documents retrieved"

**Solutions**:
1. Check documents are indexed: `python scripts/example_usage.py`
2. Verify namespace: use `--namespace general` for testing
3. Check Pinecone index: View in Pinecone console

### Issue: "Ingestion fails with Azure error"

**Solutions**:
1. Verify `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` is correct
2. Check `AZURE_DOCUMENT_INTELLIGENCE_KEY` is valid
3. Ensure endpoint has `/formrecognizer` path

### Issue: "Slow retrieval (>5 seconds)"

**Solutions**:
1. Temporarily disable reranker: `config.retrieval.use_reranker = False`
2. Reduce initial_k: `config.retrieval.initial_k = 10`
3. Check Cohere API latency

## Performance Tips

### For Development
```python
# Faster ingestion (skip LLM metadata extraction)
enricher = MetadataEnricher(use_llm=False)

# Smaller batches
config.embedding.batch_size = 50  # Default: 100

# Disable validation
python scripts/ingest_documents.py --skip-validation
```

### For Production
```python
# Enable all features
config.retrieval.use_reranker = True
config.retrieval.enable_query_expansion = True

# Optimize for accuracy
config.retrieval.initial_k = 30
config.retrieval.reranker_top_k = 10
config.retrieval.final_top_k = 5
```

## Next Steps

1. ✅ Read [ADVANCED_RAG_README.md](ADVANCED_RAG_README.md) for full documentation
2. ✅ Explore [scripts/example_usage.py](scripts/example_usage.py) for code examples
3. ✅ Customize configuration in [app/rag_config.py](app/rag_config.py)
4. ✅ Review ingestion results in `app/processed/`
5. ✅ Monitor performance with LangSmith (if enabled)

## Getting Help

- 📖 Full docs: [ADVANCED_RAG_README.md](ADVANCED_RAG_README.md)
- 🔧 Configuration: [app/rag_config.py](app/rag_config.py)
- 📝 Examples: [scripts/example_usage.py](scripts/example_usage.py)
- 📊 Logs: `app/logs/api.log`

## Summary

You now have an advanced RAG pipeline with:

| Feature | Status |
|---------|--------|
| Hierarchical chunking (parent-child) | ✅ Active |
| Cohere reranking | ✅ Active |
| Namespace organization | ✅ Active |
| Quality validation | ✅ Active |
| Nepali term mapping | ✅ Active |
| Query decomposition | ✅ Active |
| Metadata enrichment | ✅ Active |
| Azure Document Intelligence | ✅ Active |
| LangSmith tracing | ✅ Available |

**Happy querying! 🚀**
