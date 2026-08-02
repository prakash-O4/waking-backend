# PE-A — Ingestion Pipeline Research & Design

**Branch:** `pe-a/ingestion-pipeline`  
**Engineer:** Kimi  
**Type:** Research + Design only — NO implementation code changes to existing modules  
**Deliverable:** `docs/ingestion_design.md` (create it)

---

## Objective

Design the production-grade ingestion pipeline for Wakil-G, grounded in the actual data.
The stack is migrating from Supabase + Pinecone to **plain PostgreSQL + pgvector**.
Produce a design document with enough concrete detail that the next engineer can implement without ambiguity.

---

## Background: What already exists

The `app/ingestion/` module has:
- `hierarchical_chunker.py` — uses `RecursiveCharacterTextSplitter` + `MarkdownHeaderTextSplitter` from langchain. **Not structure-aware for Nepali legal text.** Must be redesigned.
- `pinecone_indexer.py` — uses Pinecone + OpenAI embeddings. **Both dependencies are being removed.**
- `metadata_enricher.py` — exists but must be reviewed against the new design.
- `document_processor.py`, `quality_validator.py` — review to understand what to keep.

The existing bitemporal PostgreSQL schema is in `migrations/001_bitemporal_schema.sql`.
Tables that already exist: `work`, `component`, `expression`, `source_publication`, `lifecycle_effect`, `precedent`, `precedent_holding`, `precedent_relation`.

The new design EXTENDS this schema — it does NOT replace or conflict with existing tables.

---

## Data sources — sample only (DO NOT read full files)

**`output/nkp_cases.jsonl`** — 1,022 Supreme Court decisions (Nepali Devanagari)
Fields: `case_id`, `decision_number`, `part`, `year_bs`, `month`, `issue_number`,
`decision_date`, `views`, `court`, `bench`, `case_number`, `case_type`,
`appellant`, `respondent`, `full_text`

Sample command (read 5 cases, inspect structure only):
```
python3 -c "
import json
with open('output/nkp_cases.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        r = json.loads(line)
        print('=== CASE', r['case_id'], '===')
        print('type:', r.get('case_type'))
        print('full_text length:', len(r.get('full_text', '')))
        print('full_text first 600 chars:')
        print(r.get('full_text','')[:600])
        print()
"
```

**`laws.jsonl`** — 677 Acts/Regulations (Nepali Devanagari)
Fields: `url`, `name`, `english_name`, `document_type`, `page`, `content`, `_id`
Note: `full_text` is empty — actual text is in `content`.

Sample command:
```
python3 -c "
import json
with open('laws.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        r = json.loads(line)
        print('=== LAW', r.get('name'), '===')
        print('type:', r.get('document_type'))
        print('content length:', len(r.get('content','')))
        print('content first 800 chars:')
        print(r.get('content','')[:800])
        print()
"
```

**`regulations.json`** — unknown schema. Sample it first:
```
python3 -c "
import json
with open('regulations.json') as f:
    data = json.load(f)
    if isinstance(data, list):
        print('list of', len(data), 'items')
        print('first item keys:', list(data[0].keys()))
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:800])
    else:
        print('type:', type(data))
        print(str(data)[:800])
"
```

**IMPORTANT:** Read only 5–10 records per file. Do not iterate the full corpus.

---

## Design tasks

### 1. Chunking strategy (the most critical output)

After sampling the data, identify the natural structure breaks in each source:

**For NKP cases:**
- Does `full_text` have recognizable Nepali section headers (facts, issues/प्रश्न, ruling/आदेश, rationale/विवेचना, headnotes/सिद्धान्त)?
- Are headers consistent across cases or variable?
- Recommend: split at section boundaries (structure-aware) vs. semantic chunking (embedding-similarity splits) vs. hybrid?
- What chunk size targets are appropriate (in characters, not tokens — Devanagari chars count differently)?

**For laws:**
- Does `content` reliably follow ऐन → परिच्छेद → दफा → उपदफा hierarchy?
- Are section markers consistent (e.g. `दफा ३.` or `३.`)?
- A proviso/स्पष्टीकरण must co-retrieve with its operative clause (PS-16). How does chunking enforce this?
- Recommend boundary rules.

**For regulations:**
- Inspect the actual structure and recommend accordingly.

**Fixed-size chunking with overlap is NOT acceptable.** The final recommendation must be one of:
- Structure-aware (regex/pattern splits at legal section markers)
- Semantic (embedding-based split points)
- Hybrid (structure-aware first, semantic for oversized sections)

Justify the choice with evidence from the sampled text.

### 2. Metadata schema per chunk

Every chunk must carry typed metadata columns (NOT a JSON blob) to enable relational filtering
without hitting the vector index.

Design the fields for each source type:

**NKP case chunks — starting schema (confirm, adjust, or add fields):**
| Field | Type | Source | Extraction method |
|---|---|---|---|
| case_id | text | record | deterministic |
| case_type | text | record | deterministic |
| court | text | record | deterministic |
| bench_type | text | record | deterministic |
| decision_date_ad | date | record | BS→AD via bs_ad_calendar.py |
| year_bs | int | record | deterministic |
| section_label | text | chunk | structure parser |
| section_type | enum | chunk | deterministic (facts/issues/ruling/rationale/order) |
| is_landmark | bool | ? | specify how to determine |
| parties_redacted | text | appellant+respondent | PII redaction (see §3) |
| cited_statutes | text[] | full_text | LLM extraction (haiku) |
| headnotes | text | full_text | LLM extraction (haiku) |
| keywords | text[] | chunk | LLM extraction (haiku) |
| relevant_questions | text[] | chunk | LLM generation (haiku, 3–5 Qs) |

**Law/Regulation chunks — starting schema:**
| Field | Type | Source | Extraction method |
|---|---|---|---|
| work_id | uuid | FK to work table | deterministic |
| act_name | text | record | deterministic |
| english_name | text | record | deterministic |
| document_type | text | record | deterministic |
| section_number | text | chunk | structure parser |
| section_title | text | chunk | structure parser |
| chapter_number | text | chunk | structure parser |
| level | enum | chunk | (act/chapter/section/subsection/proviso) |
| parent_section | text | chunk | structure parser |
| effective_date_ad | date | ? | specify source |
| keywords | text[] | chunk | LLM extraction (haiku) |
| relevant_questions | text[] | chunk | LLM generation (haiku, 3–5 Qs) |

**LLM model for all extraction:** `claude-haiku-4-5-20251001`. Specify which calls can be batched.

### 3. PII redaction for NKP cases

`appellant` and `respondent` contain real person names + addresses. `full_text` likely repeats them.

Recommend a redaction strategy given that Nepali NER tooling is immature:
- Option A: LLM-based redaction (send appellant/respondent strings to haiku, extract entity spans, replace in full_text)
- Option B: Rule-based (the appellant/respondent field values are known strings — do exact-match + fuzzy replacement in full_text)
- Option C: Hybrid (rule-based first pass, LLM second pass for variants)

For each option: what precision/recall tradeoff, what failure modes, what is your recommendation?

**Storage model:**
- Redacted text goes in the `chunks` table (public-facing)
- Original unredacted content goes in a `pii_vault` table (locked, separate access)
- The `pii_vault` must link back to the chunk/document with a stable `document_id`
- Access to `pii_vault` should require a separate PostgreSQL role

### 4. PostgreSQL schema (DDL)

Design the tables needed. Write actual DDL ready to paste into `migrations/005_ingestion_pipeline.sql`.

Required extensions: `pgcrypto`, `btree_gist` already in migration 001. Add `vector` (pgvector) here.

Tables to design:

**`documents`** — one row per source document before chunking
Must include:
- `source_type`: enum('nkp_case', 'act', 'regulation')
- `source_id`: the original ID from the JSONL (case_id or _id)
- `content_hash`: SHA-256 of raw content (idempotency key — prevents double-ingestion)
- `ingestion_status`: enum('pending', 'approved', 'rejected') — maps to PS-2 dual approval gate
- `valid_from`, `valid_to`: tstzrange (bitemporal)
- `ingested_at`: timestamptz
- `approved_by`, `second_approved_by`: text (dual approval names — PS-2)
- `raw_content`: text (immutable after write)
- Unique constraint on `(source_type, source_id)`

**`chunks`** — one row per chunk (the main retrieval unit)
Must include:
- `id`: uuid primary key
- `document_id`: uuid FK → documents
- `embedding`: vector(N) — specify dimensionality based on recommended model
- `chunk_index`: int (position within document)
- `chunk_text`: text (redacted for NKP cases)
- `chunk_type`: text (section_label)
- All typed metadata columns from §2 above (NOT a jsonb blob)
- `created_at`: timestamptz
- HNSW index on embedding (specify ef_construction and m params)
- B-tree indexes on (source_type, case_type, decision_date_ad, court)

**`pii_vault`** — unredacted NKP content
Must include:
- `id`: uuid primary key
- `document_id`: uuid FK → documents
- `appellant_raw`: text
- `respondent_raw`: text
- `full_text_raw`: text
- `stored_at`: timestamptz
- Document the required PostgreSQL role restriction as a DDL comment

**Note on embedding model:** Do NOT use OpenAI or Pinecone. Recommend a multilingual model
strong on Devanagari. Candidates: `intfloat/multilingual-e5-large` (1024-dim),
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim),
`BAAI/bge-m3` (1024-dim). State the dimensionality so the vector column size is concrete.

### 5. BM25 gap (open architectural question — research and recommend)

The system design §8 requires "BM25 Nepali" in the retrieval stack. pgvector provides only kNN.
Evaluate and recommend one of:
- **Option A:** PostgreSQL `tsvector` + `GIN` index for text search (limited Nepali stemming support — what is the actual Nepali tokenizer situation?)
- **Option B:** Keep OpenSearch for BM25, use pgvector in PostgreSQL for vectors (two systems)
- **Option C:** Use `pg_search` (ParadeDB) which provides BM25 within PostgreSQL

State tradeoffs concretely. This will be decided by Prakash before implementation.

### 6. Ingestion pipeline stages

Design the pipeline as ordered stages. For each stage: inputs, outputs, failure mode, retry policy.

```
load → validate → redact_pii → chunk → extract_metadata → embed → upsert
```

Answer:
- Which stages are parallelizable across documents?
- Which stages are parallelizable within a single document?
- Where are the rate-limit chokepoints (haiku API calls, embedding throughput)?
- How does the `content_hash` idempotency check work exactly?
- Where does the dual approval gate (PS-2) pause the pipeline?

---

## Deliverable

Create `docs/ingestion_design.md` with these sections:
1. **Data observations** — what you found in the samples (concrete, not generic)
2. **Chunking strategy per source** — with rationale tied to actual text patterns observed
3. **Metadata schema per source** — final tables
4. **PII redaction recommendation** — with tradeoffs
5. **PostgreSQL DDL** — migration 005, copy-paste ready
6. **Pipeline stage diagram** — ASCII art
7. **Open questions for Prakash** — BM25/text-search choice, embedding model choice, any other gaps

---

## Governing design references

- `SYSTEM_DESIGN.md` §2 (Core Invariants), §4 (Data model), §5 (Ingestion plane), §8 (retrieval BM25 requirement), §12 (Tech stack swappable)
- `SYSTEM_DESIGN.md` §14:
  - **PS-2** dual approval gate (ingestion_status = pending until approved)
  - **PS-3** citations to authoritative instrument chain
  - **PS-14** PII / trace redaction (pii_vault design)
  - **PS-16** provisos must co-retrieve with operative clause (chunking constraint)

## Explicitly forbidden

- Do NOT modify any existing file in `app/`, `migrations/`, `tests/`, or `scripts/`
- Do NOT implement code — design document only
- Do NOT use Pinecone, OpenAI embeddings, or Supabase client in the design
- Do NOT put all metadata in a `jsonb` blob — typed columns required
- Do NOT recommend fixed-size chunking with overlap
- Do NOT silently resolve the BM25 gap — flag it explicitly

## Required checks before committing

```bash
make lint    # must pass (no Python changes expected, but run it)
```

## Commit authorship

Every commit must use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
Never add Co-Authored-By, Generated-with, or any AI attribution line to commits.

---

## Return to Claude when done

Report:
1. Commit hash
2. Key findings from data sampling (2–3 sentences on actual text structure observed)
3. Your chunking recommendation and the primary evidence for it
4. Your BM25 recommendation
5. Any PS requirement you think the design could conflict with
