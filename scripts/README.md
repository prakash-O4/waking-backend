# Pinecone Ingestion Scripts

This directory contains scripts for ingesting and managing documents in Pinecone with unique metadata.

## Scripts

### 1. `ingest_case_law.py`
Ingests the Case Law PDF into Pinecone with unique metadata for later deletion.

**Usage:**
```bash
cd /Users/prakash/Desktop/wakilg-backend
python scripts/ingest_case_law.py
```

**What it does:**
- Loads `source/case law.pdf`
- Splits it into chunks (600 chars, 100 overlap)
- Adds unique metadata: `source_id: case_law_2024`
- Embeds and uploads to Pinecone index `wakil-g`

### 2. `delete_by_source_id.py`
Deletes all documents from Pinecone that match a specific source_id.

**Usage:**
```bash
# Delete case law documents
python scripts/delete_by_source_id.py case_law_2024

# Delete from a different index
python scripts/delete_by_source_id.py case_law_2024 --index my-index
```

**What it does:**
- Connects to Pinecone
- Filters documents by `source_id` metadata
- Deletes all matching vectors
- Asks for confirmation before deletion

## How to Ingest Case Law

1. Make sure your `.env` file has the required API keys:
   ```
   OPENAI_API_KEY=your_key
   PINECONE_API_KEY=your_key
   ```

2. Run the ingestion script:
   ```bash
   python scripts/ingest_case_law.py
   ```

3. The script will output the progress and confirm when complete.

## How to Delete Case Law

When you need to remove the case law documents (e.g., for your presentation):

```bash
python scripts/delete_by_source_id.py case_law_2024
```

Type `yes` when prompted to confirm deletion.

## Customizing Source IDs

If you need to ingest the same document multiple times with different IDs:

1. Edit `ingest_case_law.py`
2. Change the `source_id` variable (e.g., `case_law_presentation_v1`)
3. Run the script

Each ingestion with a unique `source_id` can be deleted separately.

## Technical Details

- **Chunk Size**: 600 characters
- **Chunk Overlap**: 100 characters
- **Embeddings**: OpenAI embeddings
- **Index**: wakil-g (Pinecone)
- **Metadata Fields**:
  - `source_id`: Unique identifier for this ingestion
  - `ingestion_date`: Timestamp of when it was ingested
  - Plus default metadata from PyPDFLoader (page numbers, etc.)
