# Task RET-A: Retrieval Rewrite

**Engineer:** Pi  
**Branch:** `feat/retrieval-rewrite`  
**Base:** `dev`

---

## Objective

The current `retrieve_postgres` queries the old bitemporal schema (`expression`,
`component`, `work`). Those tables are empty — all 200 ingested laws live in
`documents` and `chunks` (pgvector). Retrieval returns zero results for every
query. This task replaces the retrieval layer end-to-end.

**Pipeline after this task:**

```
question + as_of
      │
      ▼
Preprocessing (NFC · digit fold)
      │
      ▼
Query Embedding (Azure text-embedding-3-large, 1024-dim)
      │
      ▼
Eligibility Pre-filter  ← PS-6
(documents + chunks, new schema)
      │ eligible chunk_ids
      │
 ┌────┴────┐
 ▼         ▼
Vector    Lexical
ANN       tsvector GIN
 └────┬────┘
      ▼
RRF Fusion
      │
      ▼
Relevance Gate  (score threshold → abstain if nothing passes)
      │
      ▼
Cohere Rerank → top K  (opt-in: skip if COHERE_API_KEY unset)
      │
      ▼
Result Assembly
```

---

## Files in scope

| File | Action |
|---|---|
| `app/retrieval/postgres_retriever.py` | Full rewrite |
| `app/retrieval/eligibility_gate.py` | Full rewrite (new schema) |
| `app/retrieval/reranker.py` | New file |
| `app/retrieval/validation_gate.py` | Update to resolve against `chunks` |
| `app/config.py` | Add `COHERE_API_KEY: str = ""` |
| `migrations/007_retrieval_indexes.sql` | New — GIN tsvector index on chunks |
| `tests/test_eligibility_gate.py` | Rewrite for new schema |
| `tests/test_validation_gate.py` | Update for chunk-based resolution |
| `tests/test_retrieval.py` | New — retriever unit tests |

**Do NOT touch:** orchestrator, ingestion pipeline, chunkers, eval harness,
any gate logic not listed above.

---

## Schema reference

**`documents`** (the document-level record):
- `id` (uuid)
- `ingestion_status` (enum: `pending`, `approved`, `quarantined`)
- `valid_time` (tstzrange) — transaction-time window; open upper bound = still current
- `source_type`, `source_id`

**`chunks`** (the searchable units):
- `id` (uuid) — this becomes `component_uri` in hit dicts
- `document_id` (uuid) → FK to documents
- `chunk_text` (text) — this becomes `text_ne` in hit dicts
- `span_sha256` (text) — integrity hash
- `embedding` (vector 1024) — pgvector
- `chunk_type` (text)
- `effective_date_ad` (date) — legal effective date of this section; NULL = inherit from document
- `act_name` (text) — law title
- `section_number`, `section_title` (text) — for laws
- `case_id` (text) — for NKP cases
- `source_type` (enum)

---

## 1. Migration — `migrations/007_retrieval_indexes.sql`

```sql
-- GIN index for tsvector lexical search on chunk_text
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_text_fts
    ON chunks USING GIN (to_tsvector('simple', chunk_text));
```

Run via `python3 scripts/migrate.py` after applying.

---

## 2. `app/config.py`

Add one field to `Settings`:

```python
COHERE_API_KEY: str = ""
```

---

## 3. `app/retrieval/eligibility_gate.py` — full rewrite

New signature (replaces old `is_eligible(conn, component_uri, as_of)`):

```python
def eligible_chunk_ids(conn: connection, as_of: date) -> set[str]:
    """Return set of chunk UUIDs eligible for retrieval at as_of."""
```

Logic:
```sql
SELECT c.id::text
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE
    d.ingestion_status IN ('approved', 'pending')
    AND lower(d.valid_time) <= %(as_of)s::timestamptz
    AND (
        c.effective_date_ad IS NULL
        OR c.effective_date_ad <= %(as_of)s
    )
```

Returns a `set[str]` of chunk UUID strings. Called once per retrieval request,
not per chunk. Do not loop — single query.

**Remove** the old `is_eligible(conn, component_uri, as_of)` function entirely.

---

## 4. `app/retrieval/postgres_retriever.py` — full rewrite

### Function signature (unchanged — callers must not break):

```python
def retrieve_postgres(
    conn: connection, query: str, as_of: date, k: int = 5
) -> list[dict[str, Any]]:
```

### Returned hit dict format:

```python
{
    "component_uri": str(chunk.id),   # UUID string — used as evidence_id
    "text_ne": chunk.chunk_text,
    "text_hash": chunk.span_sha256,
    "score": float,                   # final score (RRF or reranker)
    "work_title_ne": chunk.act_name or chunk.case_id or "",
    "chunk_type": chunk.chunk_type,
    "section_number": chunk.section_number or "",
}
```

### Preprocessing

```python
import unicodedata

_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

def _preprocess(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_DIGIT_MAP)
```

### Query embedding

```python
from openai import AzureOpenAI
from app.config import get_settings, azure_base_url

def _embed_query(text: str) -> list[float]:
    s = get_settings()
    client = AzureOpenAI(
        api_key=s.AZURE_OPENAI_KEY,
        azure_endpoint=azure_base_url(),
        api_version=s.AZURE_OPENAI_API_VERSION,
    )
    resp = client.embeddings.create(
        model=s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text,
        dimensions=s.AZURE_OPENAI_EMBEDDING_DIMENSIONS,
    )
    return resp.data[0].embedding
```

### Eligibility pre-filter

Call `eligible_chunk_ids(conn, as_of)` once. If empty set → return `[]`
immediately (abstain — no eligible law at this as_of).

### Vector arm (ANN)

```sql
SELECT id::text, chunk_text, span_sha256, act_name, case_id,
       chunk_type, section_number,
       1 - (embedding <=> %(qvec)s::vector) AS vec_score
FROM chunks
WHERE id::text = ANY(%(eligible)s)
ORDER BY embedding <=> %(qvec)s::vector
LIMIT %(limit)s
```

`limit = k * 3`

### Lexical arm (tsvector)

```sql
SELECT id::text, chunk_text, span_sha256, act_name, case_id,
       chunk_type, section_number,
       ts_rank_cd(to_tsvector('simple', chunk_text),
                  plainto_tsquery('simple', %(query)s)) AS lex_score
FROM chunks
WHERE id::text = ANY(%(eligible)s)
  AND to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', %(query)s)
LIMIT %(limit)s
```

`limit = k * 3`

If `plainto_tsquery` produces an empty query (non-alphabetic input), skip the
lexical arm — do not error.

### RRF Fusion

```python
def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Pass `[vector_ids, lexical_ids]`. Result is `[(chunk_id, rrf_score), ...]`.

### Relevance gate

After RRF, before reranking:

```python
_RELEVANCE_THRESHOLD = 0.005  # RRF score; tune after eval
```

Filter to chunks where `rrf_score >= _RELEVANCE_THRESHOLD`. If none pass →
return `[]` (abstain). Do not retry.

### Reranker (opt-in)

```python
from app.retrieval.reranker import rerank

candidates = top k*2 after relevance gate
final = rerank(query, candidates, k)   # no-op if COHERE_API_KEY unset
```

### Result assembly

Fetch full metadata for the final chunk IDs (one `SELECT ... WHERE id = ANY(...)`)
and build the hit dicts. Preserve the reranker/RRF order.

---

## 5. `app/retrieval/reranker.py` — new file

```python
from __future__ import annotations
from typing import Any

def rerank(query: str, hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """
    Cohere reranker. Returns top-k hits re-ordered by relevance.
    No-op (returns hits[:k]) if COHERE_API_KEY is unset or cohere not installed.
    """
    from app.config import get_settings
    s = get_settings()
    if not s.COHERE_API_KEY:
        return hits[:k]
    try:
        import cohere
    except ImportError:
        return hits[:k]

    co = cohere.Client(s.COHERE_API_KEY)
    docs = [h["text_ne"] for h in hits]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs,
        top_n=k,
    )
    return [hits[r.index] for r in response.results]
```

---

## 6. `app/retrieval/validation_gate.py` — update

`validate_and_render` currently re-fetches text by `component_uri` from
`expression`. Change to re-fetch from `chunks`:

```sql
SELECT chunk_text, span_sha256
FROM chunks
WHERE id = %(component_uri)s::uuid
```

Replace `text_hash` check with `span_sha256` check. All other logic (hash
comparison, abstain path) stays identical.

Also update the eligibility re-check inside validation gate: instead of calling
the old `is_eligible(conn, component_uri, as_of)`, call:

```python
eligible_chunk_ids(conn, as_of)  # and check component_uri in that set
```

Or inline an equivalent single-row eligibility query — your choice, but the
check must still happen.

---

## 7. Tests

### `tests/test_eligibility_gate.py` — rewrite

Test `eligible_chunk_ids` with a mocked cursor. Scenarios:
- document `pending` with `valid_time` containing `as_of` → chunk included
- document `quarantined` → chunk excluded
- `effective_date_ad > as_of` → chunk excluded
- empty result → returns empty set

### `tests/test_validation_gate.py` — update

Mock `chunks` query instead of `expression`. Same logic, new table.

### `tests/test_retrieval.py` — new

Mock `eligible_chunk_ids`, `_embed_query`, and the DB cursor. Test:
- empty eligible set → returns `[]`
- vector arm results only (lexical empty)
- both arms → RRF merges correctly
- relevance gate: all below threshold → `[]`
- reranker skipped when `COHERE_API_KEY` unset

---

## Required checks

```
make test
make lint
```

Test count will change: `test_eligibility_gate.py` and `test_validation_gate.py`
are being rewritten; `test_retrieval.py` is new. Final count will be higher than
33. All must pass.

Return commit hash, changed files, checks run/results.

---

## What does NOT change

- `gated_orchestrator.py` — calls `retrieve_postgres(conn, q, as_of)` unchanged
- All ingestion pipeline files
- All eval harness and golden sets
- Zero-tolerance gates: `repealed-as-current=0`, `not-yet-effective-as-current=0`

---

## PS requirements in scope

- **PS-6** — eligibility gate on every retrieval path; temporal validity per chunk
- **PS-7** — abstention is server-owned; relevance gate returns `[]`, not a retry
- **PS-12** — retrieved text remains untrusted (no change to how context is wrapped)

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution.
