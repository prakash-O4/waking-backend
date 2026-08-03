# PE-A — Ingestion Pipeline Implementation

**Branch:** `pe-a/ingestion-pipeline`
**Engineer:** Kimi
**Governing design:** `docs/ingestion_design.md` (approved; read it fully before writing code)
**Decisions locked in:** pg_search (ParadeDB) for BM25 · bge-m3 for embeddings · regulations out of scope (PE-B)

---

## Objective

Implement the ingestion pipeline described in `docs/ingestion_design.md`.
Replace the Pinecone + OpenAI stack with PostgreSQL + pgvector + bge-m3.
Write the pipeline as a series of well-separated, testable stages.

---

## Files to CREATE (new)

| File | Purpose |
|---|---|
| `migrations/005_ingestion_pipeline.sql` | DDL: documents, chunks, pii_vault, pg_search index — copy the DDL from §5 of the design doc verbatim, then add pg_search extension + BM25 index (see below) |
| `app/ingestion/laws_chunker.py` | Structure-aware chunker for acts (дафа-anchor splits; see §2.1 of design) |
| `app/ingestion/nkp_chunker.py` | Hybrid chunker for NKP cases (§2.2 of design) |
| `app/ingestion/pii_redactor.py` | Hybrid PII redaction for NKP (§4 of design) |
| `app/ingestion/pgvector_indexer.py` | pgvector upsert (replaces pinecone_indexer.py; see spec below) |
| `app/ingestion/pipeline.py` | 8-stage pipeline orchestrator (§6 of design) |
| `scripts/ingest_nkp.py` | CLI script: loads output/nkp_cases.jsonl through the pipeline |
| `tests/test_ingestion_pipeline.py` | Unit tests (see test spec below) |

## Files to REWRITE (keep filename, replace content)

| File | Change |
|---|---|
| `app/ingestion/metadata_enricher.py` | Remove langchain_openai / ChatOpenAI; use Anthropic haiku-4-5 SDK directly; update output schema to match typed columns (see §3 of design) |
| `scripts/ingest_laws.py` | Remove Pinecone/Supabase path; wire through new pipeline |
| `app/ingestion/__init__.py` | Remove HierarchicalChunker, PineconeIndexer; export LawsChunker, NKPChunker, PIIRedactor, PgvectorIndexer, IngestionPipeline |

## Files to DELETE

- `app/ingestion/pinecone_indexer.py` — replaced by pgvector_indexer.py
- `app/ingestion/hierarchical_chunker.py` — replaced by laws_chunker.py + nkp_chunker.py

## Files to touch ONLY to fix broken imports (no logic changes)

- `app/ingestion/quality_validator.py` — imports HierarchicalChunk from hierarchical_chunker; remove that import, keep all other logic intact
- `scripts/ingest_documents.py` — PDF pipeline (PE-B scope); just fix the import of HierarchicalChunker/PineconeIndexer so it doesn't crash at startup; do NOT change its logic

## Files NOT to touch

Everything in `app/retrieval/`, `app/authority/`, `app/main.py`, `app/eval/`, `tests/` except the new test file. `app/ingestion/document_processor.py` stays (PDF, PE-B). `migrations/001–004` stay.

---

## Migration 005 spec

Start with the DDL block from `docs/ingestion_design.md §5` verbatim (the CREATE EXTENSION vector, enums, documents, chunks, pii_vault, indexes, role grant). Then append:

```sql
-- pg_search (ParadeDB): BM25 full-text index on chunks.
-- REQUIRES: ParadeDB extension installed on the PostgreSQL host.
-- If unavailable, comment this out and fall back to GIN tsvector (Option A).
CREATE EXTENSION IF NOT EXISTS pg_search;

-- BM25 index for Nepali text search. ParadeDB tokenizes on unicode word boundaries
-- (no Nepali stemmer — morphological variants handled at query time via variant expansion).
-- Consult the ParadeDB version installed for exact index syntax; the form below is
-- correct for ParadeDB ≥0.8.
CALL paradedb.create_bm25(
    index_name => 'chunks_bm25',
    table_name => 'chunks',
    key_field  => 'id',
    text_fields => paradedb.field('chunk_text', tokenizer => paradedb.tokenizer('unicode'))
);
```

If the `paradedb.create_bm25` syntax differs for the installed version, look it up from
`SELECT paradedb.schema_version()` at runtime and adjust. The comment in the migration
must document the version assumption.

---

## laws_chunker.py spec

```python
# Input: one record from laws.jsonl (dict with 'content', 'name', 'english_name', '_id', 'document_type')
# Output: list of LawChunk dataclasses

@dataclass
class LawChunk:
    chunk_index: int
    chunk_text: str         # verbatim from content (including <amend> tags)
    embed_text: str         # <amend>…</amend> and ✂ stripped, for embedding only
    level: str              # 'act' | 'chapter' | 'section' | 'subsection' | 'proviso'
    section_number: str | None
    section_title: str | None
    chapter_number: str | None
    parent_section: str | None   # दफा number when level='subsection'/'proviso'
    co_retrieve_parent_index: int | None  # chunk_index of operative clause (PS-16)
```

Split rules (from design §2.1):
1. Extract preamble/header block (everything before first `**[०-९]` bold heading) → one chunk, `level='act'`
2. Split on `(?=\*\*[०-९]+\.)` to get दफा chunks; track enclosing `## परिच्छेद-N` for chapter_number
3. If दफा chunk > 2,400 chars, split at उपदफा `^\([०-९]+\)` boundaries; each piece is `level='subsection'`, `parent_section=दफा_number`
4. स्पष्टीकरण blocks (`स्पष्टीकरण\s*[:：]` line) within a chunk: if within size limit, keep inline (don't orphan). If the containing दफा was split, create a `level='proviso'` chunk with `co_retrieve_parent_index` pointing at the operative subsection chunk_index.
5. After splitting, normalize Devanagari (unicodedata.normalize('NFC', text)) on chunk_text.
6. embed_text = chunk_text with `<amend>[^<]+</amend>` replaced by `[संशोधित]` and `✂+\.+` replaced by `[…]`.
7. Emit chunks in document order; chunk_index is 0-based.

Minimum chunk size: 400 chars. If a दफा is < 400 chars and has a sibling दफा, do NOT merge — emit as-is (legal units must not be merged across boundaries).

---

## nkp_chunker.py spec

```python
@dataclass
class NKPChunk:
    chunk_index: int
    chunk_text: str          # redacted text
    embed_text: str          # same as chunk_text (NKP has no markup to strip)
    section_type: str        # caption | headnote | advocates | opinion | order | colophon | body
    section_label: str       # human-readable label
```

Primary split anchors (apply in order; each match consumes the text up to the next anchor):

| Pattern | section_type | Notes |
|---|---|---|
| `^सर्वोच्च अदालत` to end of judge-name lines | `caption` | First lines of document |
| `^आदेश मिति\s*:` or `^मुद्दाः` or `^विषयः` lines | `caption` | Optional; merge into caption if adjacent |
| Text before first `का तर्फबाट` / `न्या\.` line | `headnote` | सिद्धान्त statements |
| Lines matching `[^।]+का तर्फबाट\s*:` | `advocates` | All advocate lines as one chunk |
| `न्या\.[^\s]+\s*:` opener through start of order block | `opinion` | Longest section |
| Line-start `^[०-९]+\.` numbered items (the tail) | `order` | Dispositive section |
| `^इति संवत्` | `colophon` | Document terminator |

Anything not matched: `section_type='body'`, do not silently drop.

Secondary split (for chunks > 2,400 chars — mainly `opinion`):
1. Split at line-start `^[०-९]+\.` (numbered items within opinion)
2. Else split at `\s। \s` (sentence boundary on Nepali danda)
3. Never split mid-word. Each piece inherits parent's section_type.

chunk_text is the redacted text (redaction happens BEFORE chunking in the pipeline — see pipeline.py).

---

## pii_redactor.py spec

```python
class PIIRedactor:
    def redact(self, full_text: str, appellant: str, respondent: str) -> tuple[str, list[str]]:
        """
        Returns: (redacted_text, list_of_redaction_warnings)
        Raises: RedactionVerificationError if any raw token ≥4 chars survives in output.
        """
```

Stage 1 — deterministic (rule-based):
1. Normalize appellant and respondent to NFC + digit-fold (replace Devanagari digits with ASCII equivalents for comparison only; the stored text keeps original).
2. Split each into tokens on spaces/punctuation.
3. Build a set of tokens ≥4 chars from both strings.
4. Exact-string replace the full `appellant` string and the full `respondent` string with `[[वादी]]` and `[[प्रतिवादी]]` respectively.
5. Also replace each individual token ≥4 chars (Devanagari word boundary `\b` doesn't work for Devanagari — use `(?<![^\s।,।])token(?![^\s।,।])` anchors or just replace all occurrences).

Stage 2 — haiku second pass:
- Only run if Stage 1 detected > 0 replacements in the text (if no replacements, Stage 1 already confirmed no PII).
- Call `claude-haiku-4-5-20251001` with a prompt like: "Given these party names: [appellant], [respondent] — identify any variant mentions (abbreviated names, honorifics, partial names) in the following text and return a JSON list of spans to replace. Format: [{\"original\": \"...\", \"replacement\": \"[[वादी]]\"}]". Apply the replacements.
- This call is per-document (not batched), with a 30-second timeout.

Stage 3 — verification assertion:
- For every token ≥4 chars from appellant/respondent, assert it does not appear in redacted_text.
- If assertion fails: raise `RedactionVerificationError(token=..., document_id=...)`. The pipeline catches this and sets `ingestion_status='pending'` with a flag — the document is quarantined for human review, never silently passed.

```python
class RedactionVerificationError(Exception):
    def __init__(self, token: str, document_id: str): ...
```

---

## pgvector_indexer.py spec

```python
class PgvectorIndexer:
    def __init__(self, conn: psycopg2.connection): ...

    def embed_chunks(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Load BAAI/bge-m3 once (cache on self), encode texts in batches, normalize."""

    def upsert_document(self, document: dict, chunks: list[LawChunk | NKPChunk], embeddings: list[list[float]]) -> str:
        """
        Insert documents row + all chunk rows in one transaction.
        Returns document_id (UUID).
        Inserts chunks in chunk_index order (0, 1, 2 …) to respect co_retrieve_parent_id FK.
        On UNIQUE(document_id, chunk_index) conflict: DELETE existing chunks for this document, re-insert.
        """

    def insert_pii_vault(self, document_id: str, appellant: str, respondent: str, full_text_raw: str) -> None:
        """Write to pii_vault. Called only for nkp_case source_type."""
```

Use `psycopg2-binary` (already in requirements.txt) and `pgvector` Python package for the vector column type. Register the pgvector adapter with `register_vector(conn)` from `pgvector.psycopg2`.

Load bge-m3:
```python
from sentence_transformers import SentenceTransformer
self._model = SentenceTransformer('BAAI/bge-m3')
```
Normalize embeddings (`normalize_embeddings=True`). Dimensionality: 1024.

---

## metadata_enricher.py spec (rewrite)

Remove all `langchain_openai` / `ChatOpenAI` imports. Use `anthropic` SDK directly.

```python
import anthropic

_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

def enrich_law_chunks(act_record: dict, chunks: list[LawChunk]) -> list[dict]:
    """
    One haiku call per act: send numbered chunk list, get back keywords + relevant_questions per chunk.
    Returns list of metadata dicts aligned by chunk_index.
    """

def enrich_nkp_chunks(case_record: dict, chunks: list[NKPChunk]) -> list[dict]:
    """
    Two haiku calls per case:
    1. cited_statutes + headnotes cleanup over full redacted text.
    2. keywords + relevant_questions per chunk (batched).
    """
```

Use `claude-haiku-4-5-20251001`. Parse all LLM outputs as JSON. If JSON parsing fails, log a warning and return empty values (never crash the pipeline; NULL columns are acceptable).

For the Message Batches API (offline ingestion — 50% cost):
Use `_client.messages.batches.create(requests=[...])` if the batch covers ≥10 documents.
For smaller runs, use individual `_client.messages.create(...)` calls.

---

## pipeline.py spec

```python
class IngestionPipeline:
    def ingest_law(self, record: dict) -> str | None:
        """Run stages 1–8 for a laws.jsonl record. Returns document_id or None if skipped."""

    def ingest_nkp_case(self, record: dict) -> str | None:
        """Run stages 1–8 for an nkp_cases.jsonl record."""
```

Stages (from design §6):
1. **LOAD**: compute `content_hash = hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()`. Check `documents` table for `(source_type, source_id)`. If exists + same hash → return None (skip). If exists + different hash → update `valid_time` upper bound on the old row, insert new row `status='pending'`. If new → insert `status='pending'`.
2. **VALIDATE**: check content not empty; for laws, verify at least one `**[०-९]` heading found. Else set `status='rejected'`, return None.
3. **REDACT_PII** (nkp_case only): call `PIIRedactor.redact()`. On `RedactionVerificationError`: log error, leave `status='pending'` with a `redaction_failed=True` flag (add this boolean column to documents in the migration), return None.
4. **CHUNK**: call LawsChunker or NKPChunker depending on source_type.
5. **EXTRACT_METADATA**: call MetadataEnricher; retry haiku calls up to 3× with exponential backoff on 429/5xx; on persistent failure, leave LLM columns NULL.
6. **EMBED**: call `PgvectorIndexer.embed_chunks()` on `embed_text` list. Batch size 32. On OOM, halve batch size and retry ×3.
7. **UPSERT**: call `PgvectorIndexer.upsert_document()` and (for NKP) `insert_pii_vault()`.
8. **DUAL APPROVAL PAUSE**: the document is now `status='pending'` in the DB. Do NOT set `status='approved'` in the pipeline. Log "document {document_id} awaiting dual approval." This is where humans intervene via a separate admin path.

The pipeline must never crash on a single document's failure — catch per-document exceptions, log them with the source_id, and continue to the next document.

---

## scripts/ingest_laws.py spec (rewrite)

```python
# Usage: python scripts/ingest_laws.py --input laws.jsonl [--limit N] [--dry-run]
# Iterates laws.jsonl line by line, calls pipeline.ingest_law(record) for each.
# Reports: processed / skipped / rejected / failed counts.
```

---

## scripts/ingest_nkp.py (new)

```python
# Usage: python scripts/ingest_nkp.py --input output/nkp_cases.jsonl [--limit N] [--dry-run]
# Iterates nkp_cases.jsonl, calls pipeline.ingest_nkp_case(record) for each.
# Reports same counts.
```

---

## requirements.txt changes

Add only these three lines (do NOT remove anything — other packages may be used by retrieval-side code):
```
anthropic>=0.40.0
sentence-transformers>=3.0.0
pgvector>=0.3.0
```

Do NOT remove `langchain-openai`, `pinecone`, `supabase`, or `chromadb` — check imports before touching anything. If a package is only imported in the two files being deleted (pinecone_indexer.py, hierarchical_chunker.py), note it in your return report but do NOT remove it from requirements.txt in this PR (that's a separate cleanup).

---

## Tests (tests/test_ingestion_pipeline.py)

Write unit tests for the following (mock DB and haiku calls; do NOT hit real APIs in tests):

1. **LawsChunker**: feed a 3-दफा act fixture (inline string), assert chunk count, levels, co_retrieve links, and that no दफा boundary is crossed mid-chunk.
2. **NKPChunker**: feed a minimal NKP case fixture, assert caption/headnote/order sections are correctly identified.
3. **PIIRedactor**: assert Stage 1 replaces the exact appellant/respondent strings; assert Stage 3 raises `RedactionVerificationError` if a token ≥4 chars survives.
4. **Pipeline idempotency**: mock the DB to return an existing row with the same content_hash; assert `ingest_law()` returns None without calling embed or upsert.
5. **Dual approval gate (DDL)**: run the migration SQL against a test SQLite-like fixture — actually, use psycopg2 with a real local DB if available; otherwise just parse the CHECK constraint logic and assert it would reject single-approver approval. Note if skipped.

Run with `make test`. All 5 tests must pass.

---

## BS→AD conversion

Use the existing `app.authority.bs_ad_calendar` module (Phase PC-A implementation). Do NOT use any date library.

```python
from app.authority.bs_ad_calendar import lookup, BeyondCalendarRange

def parse_decision_date(decision_date_bs: str) -> tuple[date | None, bool]:
    """
    Parses '२०८१/०१/०३' (Devanagari digits, BS) to AD date.
    Returns (ad_date, is_boundary_window).
    Returns (None, False) if unparseable or out of calendar range (log warning).
    """
    def devanagari_to_int(s: str) -> int:
        return int(''.join(str(ord(c) - ord('०')) for c in s))
    ...
```

---

## Commit authorship

Every commit must use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
Never add Co-Authored-By or any AI attribution line.

---

## Required checks before reporting back

```bash
make lint    # ruff + mypy — must be clean
make test    # all tests green
```

---

## Return to Claude when done

Report:
1. Commit hash
2. Files created / modified / deleted
3. `make lint` and `make test` output (or paste the summary)
4. Any assumption you made that wasn't in this brief
5. Any PS-* requirement you think your implementation might be touching unexpectedly
