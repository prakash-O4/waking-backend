# Ingestion Pipeline — Research & Design (PE-A)

**Branch:** `pe-a/ingestion-pipeline`
**Scope:** design only. No implementation in this change. Target migration: `migrations/005_ingestion_pipeline.sql` (DDL in §5, copy-paste ready).
**Governing refs:** `system-design.md` §2 (invariants), §4 (data model), §5 (ingestion plane), §8 (retrieval), §14 (PS-2/PS-3/PS-14/PS-16).

---

## 1. Data observations

Sampled 6 NKP cases, 6 laws, and the full top-level structure of `regulations.json` (5–10 records per file, per the brief — no full-corpus iteration).

### 1.1 `output/nkp_cases.jsonl` (Supreme Court decisions)

- All 6 sampled cases are **संयुक्त इजलास** (division bench), 20k–44k chars of `full_text`. `decision_date` is BS (`२०८१/०१/०३`); `year_bs` in the sample was uniformly `२०८२` even for २०८०/२०८१ decisions — **do not trust `year_bs` as decision year; derive it from `decision_date`**.
- Consistent skeleton across all samples:
  1. Caption block: `सर्वोच्च अदालत, संयुक्त इजलास` + `माननीय न्यायाधीश श्री …` lines.
  2. Optional `आदेश मिति : …` and `मुद्दाः …` / `विषयः …` lines (present in 5/6).
  3. **Headnote paragraphs**: unnumbered सिद्धान्त statements before any advocate names (present in all 6).
  4. Advocate lines: `निवेदकका तर्फबाट : …`, `प्रत्यर्थीका तर्फबाट : …` (regex `का तर्फबाट` hits in all 6).
  5. Judge-attributed opinion: `न्या.<name> :` opens the reasoning (hit in 4/6; the others use equivalent unmarked narration).
  6. **Numbered order items**: `१.` … `९.` at line start (8–18 items per case) — the dispositive/operative part.
  7. Colophon: `इति संवत् … गते रोज … शुभम्` (present in all 6) — a reliable document terminator.
- Section headers are **not standardized**: keyword counts for प्रश्न/विवेचना/सिद्धान्त/फैसला vary wildly (e.g. case 10479 has 17 प्रश्न hits, case 10476 has 0). There is no uniform "facts / issues / ruling" header vocabulary to split on. Any NKP chunker keyed on fixed Nepali headers will silently degrade to one giant chunk on a large fraction of the corpus.
- `appellant`/`respondent` contain full name-plus-address strings, e.g. `पर्सा जिल्ला, … वडा नं.१४ स्थायी घर भई हाल … बस्ने सुरेशकुमार शुक्ला`. Token-level verbatim match against `full_text`: 3/4 to 13/14 of appellant tokens appear in the text — i.e. **most but not all** PII strings repeat verbatim in the body (see §4).

### 1.2 `laws.jsonl` (Acts)

- `full_text` is empty; text is in `content`. Structure is **semi-markdown and highly regular**:
  - दफा headings are bold numbered: `**१. संक्षिप्त नाम र प्रारम्भ:**` — a clean split anchor (present in all 6 samples; 9–47 दफा each).
  - परिच्छेद headers appear as `## परिच्छेद-१` markdown where the act has chapters (2 of 6 samples; others have no परिच्छेद level).
  - उपदफा are Devanagari-numeral parenthesized markers `(१)` `(२)` — 15–168 per act.
  - Provisos/explanations appear as `स्पष्टीकरण : …` / `स्पष्टीकरणः …` blocks and as inline proviso sentences — 0–5 explicit स्पष्टीकरण per act.
  - **Amendment markup is inline**: `<amend>…द्वारा संशोधित ।</amend>` tags (up to 25 per act) and `✂.....` elision marks. The `content` is a **consolidation, not the original gazette text** — per PS-3 it must be stored as a derived source kind, never as `official_original`.
  - Header block carries `प्रमाणीकरण र प्रकाशन मिति` (BS) and a `संशोधन गर्ने ऐन` amendment list — the raw material for `lifecycle_effect` proposals.
- Section-size distribution (split on `**N.` headings, 6 acts): min ~90 chars, **median 350–720 chars, max ~1,800–4,000 chars**. So दफा-level chunks fit comfortably except a long tail (up to ~4k chars) that must split at उपदफा boundaries.
- **Commencement is not uniform**: अनुगमन तथा मूल्याङ्कन ऐन, २०८० दफा १(२) reads `यो ऐन प्रमाणीकरण भएको एकतिसौँ दिनदेखि प्रारम्भ हुनेछ` — a delayed commencement, and other instruments use the `राजपत्रमा सूचना` pattern (PS-2). `effective_date_ad` must therefore come from approved `lifecycle_effect` commence rows, never from a parser guess of the प्रमाणीकरण date.

### 1.3 `regulations.json`

- **Contains no regulation text.** It is a dict of 24 खण्ड (subject categories) → lists of `{title, url, id, popularity_factor, english_name}`, plus a `_backend_only` key. The `url` fields point to PDF uploads.
- Consequence: regulation ingestion is a **fetch → PDF→markdown → parse** pipeline (the existing `app/utils/pdf_to_markdown.py` / Azure Document Intelligence path in `document_processor.py` is the fetch target), after which regulations look structurally like acts (नियमावली use the same दफा/उपदफा conventions). This is an open dependency, flagged in §7.

---

## 2. Chunking strategy per source

**Global rule (per brief): no fixed-size chunking with overlap anywhere.** Target chunk size where splitting is needed: **~1,800 chars target, 2,400 chars hard max, 400 chars soft min** (Devanagari chars; ~1.5–2.5 chars/token for the candidate multilingual embedding tokenizers, so 2,400 chars ≈ 1,000–1,600 tokens — well inside the 8,192-token context of the recommended models, leaving room for context-assembly prefixes).

### 2.1 Laws (`act`) — structure-aware

Split at the anchors observed in §1.2:

1. **दफा boundary**: split on `(?=\*\*[०-९]+\.)` headings. Each दफा = one candidate chunk, carrying `chapter_number` from the enclosing `## परिच्छेद-N` (nullable — most sampled acts have no परिच्छेद).
2. **Oversized दफा (>2,400 chars)**: split at उपदफा boundaries `(१)` `(२)` …; each piece keeps `parent_section` = the दफा number and `level = 'subsection'`. Never split inside an उपदफा; if a single उपदफा exceeds the max (rare; not observed in samples), split at paragraph boundaries with `parent_section` retained.
3. **Proviso/स्पष्टीकरण co-retrieval (PS-16)**: a स्पष्टीकरण block or proviso sentence is **never emitted as a standalone top-level chunk**. If the operative दफा fits in one chunk, the proviso stays inside it (the common case — median 350–720 chars). If the दफा was split at उपदफा boundaries, the proviso chunk gets `level = 'proviso'` and `co_retrieve_parent_id` pointing at the operative subsection chunk (and transitively the दफा). Context assembly on the query side must **always fetch `co_retrieve_parent_id` chains** — this is the deterministic enforcement of PS-16, eval-asserted.
4. **Preamble/header block** (title, प्रमाणीकरण मिति, संशोधन सूची, प्रस्तावना) is parsed into metadata (`documents` row + `lifecycle_effect` proposals) and also kept as one `level = 'act'` header chunk for citation context.
5. `<amend>…</amend>` tags and `✂` elisions are **preserved verbatim in `chunk_text`** (they are provenance, PS-10) but stripped from the text fed to the embedding model, with offsets recorded so the span hash still matches the stored text.

**Rationale:** the दफा anchors were present and regular in 6/6 samples; median दफा size (~700 chars) is near-ideal retrieval granularity; semantic chunking would add cost and nondeterminism where a deterministic parser already gives legally meaningful units. Fixed-size splitting would sever provisos from operative clauses — the exact PS-16 failure.

### 2.2 NKP cases — hybrid (structure-aware first, size-driven secondary split)

Primary splits on the deterministic anchors observed in §1.1, in order:

| Priority | Anchor (regex) | `section_type` |
|---|---|---|
| 1 | Caption block `सर्वोच्च अदालत,…इजलास` + justice lines | `caption` |
| 2 | `आदेश मिति` / `मुद्दाः` / `विषयः` lines | `caption` |
| 3 | Headnote paragraphs (text before first `का तर्फबाट` or `न्या.` line) | `headnote` |
| 4 | `…का तर्फबाट : …` lines | `advocates` |
| 5 | `न्या\.<name> :` opener through the start of the final order block | `opinion` |
| 6 | Line-start numbered items `१.` … at the tail (the dispositive list) | `order` |
| 7 | `इति संवत् … शुभम्` block | `colophon` |

Anything that fails to match (variable headers, §1.1) falls into `section_type = 'body'` rather than being force-fit.

**Secondary split (the "hybrid" part):** any chunk >2,400 chars — typically `opinion` (opinions run 15k–35k chars in the samples) — is split **at line-start numbered-item boundaries (`^[०-९]+\.`)** first, then at paragraph boundaries, then (last resort) at sentence boundaries on ` । `, never mid-sentence. Every secondary piece inherits the parent's `section_type` and gets sequential `chunk_index`. Semantic (embedding-similarity) splitting was considered and rejected: it is nondeterministic across embedding-model versions, which breaks re-ingestion diffing and span-hash stability (§7.7 of the system design); numbered-item boundaries already track the opinion's issue-by-issue structure.

**Rationale:** the anchors above fired on all 6 samples (caption, headnotes, colophon 6/6; advocates 6/6; judge attribution 4/6), and the numbered tail is the dispositive part users cite. But header vocabulary is too inconsistent for a pure header-keyed splitter, so the size-driven fallback is mandatory, not optional.

### 2.3 Regulations — same chunker as acts, after a fetch + PDF→markdown stage

`regulations.json` supplies only metadata + PDF URLs (§1.3). After download and PDF→markdown conversion, नियमावली follow the same दफा/उपदफा/परिच्छेद conventions, so the §2.1 chunker applies unchanged, with `source_type = 'regulation'`. OCR confidence from the converter must flow into `documents` (PS-10 badge) — this path is OCR-derived where acts were born-digital.

---

## 3. Metadata schema per chunk (typed columns — no jsonb blob)

LLM model for all extraction: **`claude-haiku-4-5-20251001`**.

### 3.1 NKP case chunks

| Field | Type | Source | Extraction method |
|---|---|---|---|
| case_id | text | record | deterministic |
| case_type | text | record | deterministic (trimmed; samples include leading `- ` artifacts) |
| court | text | record | deterministic |
| bench_type | text | record `bench` | deterministic (e.g. संयुक्त इजलास) |
| decision_date_ad | date | record `decision_date` (BS) | BS→AD via `bs_ad_calendar` table lookup — **never a library call** (PS-5); boundary-window dates go to human review |
| year_bs | int | derived | from `decision_date`, not the record's `year_bs` field (see §1.1) |
| section_label | text | chunk | structure parser (§2.2) |
| section_type | enum | chunk | deterministic: `caption/headnote/advocates/facts/issues/opinion/order/colophon/body` |
| is_landmark | bool | derived | initial heuristic: `bench_type` ∈ {पूर्ण इजलास, संवैधानिक इजलास} OR headnote count ≥ 2; refined later from approved `precedent_relation` counts (§7 open question) |
| parties_redacted | text | appellant + respondent | after §4 redaction |
| cited_statutes | text[] | full_text | LLM extraction (haiku), one call per case, validated against `work.title_ne` |
| headnotes | text | full_text | deterministic (headnote block, §2.2) with haiku cleanup pass |
| keywords | text[] | chunk | LLM extraction (haiku) |
| relevant_questions | text[] | chunk | LLM generation (haiku, 3–5 Qs) |

### 3.2 Law/Regulation chunks

| Field | Type | Source | Extraction method |
|---|---|---|---|
| work_id | uuid | FK → `work` | deterministic (work upserted during document load) |
| act_name | text | record `name` | deterministic |
| english_name | text | record | deterministic |
| document_type | text | record | deterministic (`act` / `regulation`) |
| section_number | text | chunk | structure parser (दफा N) |
| section_title | text | chunk | structure parser (heading text) |
| chapter_number | text | chunk | structure parser (परिच्छेद, nullable) |
| level | enum | chunk | `act/chapter/section/subsection/proviso` |
| parent_section | text | chunk | structure parser (NULL for top-level दफा) |
| effective_date_ad | date | `lifecycle_effect` (commence, approved) | **join at write time; NULL-pending for unverified commencement** (PS-2 — never fabricated) |
| keywords | text[] | chunk | LLM extraction (haiku) |
| relevant_questions | text[] | chunk | LLM generation (haiku, 3–5 Qs) |

### 3.3 LLM batching plan

- **Batchable**: `keywords` + `relevant_questions` for all chunks of one document in a single haiku call (input = numbered chunk list, output = JSON array keyed by chunk index). Across documents, use the Anthropic Message Batches API (50% cost, 24h SLA is fine for offline ingestion).
- **Per-document single calls**: `cited_statutes` + `headnotes` cleanup for NKP (one call per case over the full text, ~30k chars — within haiku context).
- **Not batched**: PII second pass (§4) — per-case, because it needs the case's own entity list as context.
- All haiku outputs are **proposals**: `cited_statutes` is validated against the `work` table and dropped if unmatched; extraction failures leave columns NULL, never guessed.

---

## 4. PII redaction recommendation (NKP cases)

**Recommendation: Option C — hybrid (rule-based first pass, haiku second pass).**

- **Option A (LLM-only)**: best recall on name variants, but sends unredacted PII to a third-party API by design, costs per-case calls over ~30k-char texts, and — worse — its span output is nondeterministic, so re-runs can produce different redactions of the same document, breaking `content_hash`-based idempotency auditing. Precision good, failure mode = silent misses with no way to enumerate them.
- **Option B (rule-only)**: fully deterministic, auditable, zero API exposure. But measured verbatim-token recall on the samples was **imperfect** (§1.1: as low as 3/4 appellant tokens and 2/10 respondent tokens matching — the rest are spelling variants, honorifics, or name-only mentions without the address prefix). Alone it leaks exactly the casual name mentions a reader actually remembers.
- **Option C (hybrid)**: deterministic pass first — exact-match the full `appellant`/`respondent` strings, then fuzzy token-sequence match (Devanagari-normalized, NFC + digit-fold, allowing honorifics श्री/विद्वान् and token reordering) — replaced with stable placeholders `[[वादी]]` / `[[प्रतिवादी]]`. Then one haiku pass per case with the known party strings as context to catch residual variants; its replacements are also placeholder-stable (same tokens), so output stays deterministic *in form* even if recall varies. Failure mode: LLM pass misses a rare variant → mitigated by the fact that rule pass already removed every structured occurrence (caption, advocate lines, order items are template positions and match reliably).
- **Verification**: post-redaction assertion — none of the raw party tokens ≥4 chars may appear in any stored `chunk_text`; failures quarantine the document (`ingestion_status` stays `pending`, flagged for human review).

**Storage model (PS-14):**

- Redacted text → `chunks.chunk_text` (the only text the retrieval path ever sees).
- Original `full_text`, `appellant`, `respondent` → `pii_vault` (one row per document, FK `document_id`).
- `pii_vault` is readable only by a dedicated PostgreSQL role (`pii_vault_reader`); the application role gets **no grant** (DDL in §5). This mirrors the system design's §11 "raw content = named roles, audited, dual-control."
- `documents.raw_content` stores the **redacted** canonical text for acts/regulations; for NKP cases it stores the redacted full text too — unredacted NKP content exists *only* in `pii_vault`.

---

## 5. PostgreSQL DDL — `migrations/005_ingestion_pipeline.sql` (copy-paste ready)

Extends (never alters) the schema from migrations 001–004. `pgcrypto` and `btree_gist` already exist; this adds `vector`.

```sql
-- migrations/005_ingestion_pipeline.sql
-- PE-A ingestion pipeline: documents, chunks (search derivative), pii_vault.
-- Design: docs/ingestion_design.md. Extends migrations 001–004; replaces nothing.
--
-- NOTE (system-design §2 invariant 1): `chunks` and its embedding index are a
-- SEARCH DERIVATIVE. The bitemporal tables (work/component/expression/
-- lifecycle_effect/precedent*) remain the only authority. Every citation is
-- revalidated against the authority store before it ships.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE ingestion_source_type AS ENUM ('nkp_case', 'act', 'regulation');
CREATE TYPE ingestion_status      AS ENUM ('pending', 'approved', 'rejected');
CREATE TYPE case_section_type     AS ENUM (
    'caption', 'headnote', 'advocates', 'facts',
    'issues', 'opinion', 'order', 'colophon', 'body'
);
CREATE TYPE law_level             AS ENUM (
    'act', 'chapter', 'section', 'subsection', 'proviso'
);

-- ---------------------------------------------------------------------------
-- documents: one row per source document, before chunking.
-- Ingestion of legal state is human-gated with dual approval (PS-2 / §2.5):
-- a document is searchable only when ingestion_status = 'approved', which the
-- CHECK below ties to two distinct named approvers.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type        ingestion_source_type NOT NULL,
    source_id          TEXT NOT NULL,              -- case_id or laws._id / regulations id
    content_hash       TEXT NOT NULL,              -- SHA-256 of normalized raw content (idempotency key)
    ingestion_status   ingestion_status NOT NULL DEFAULT 'pending',
    valid_time         TSTZRANGE NOT NULL DEFAULT tstzrange(now(), NULL),  -- (valid_from, valid_to) bitemporal
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_by        TEXT,                       -- Gatekeeper (first approver)
    second_approved_by TEXT,                       -- second approver; must differ
    raw_content        TEXT NOT NULL,              -- immutable after write; REDACTED for nkp_case
    ocr_confidence     FLOAT,                      -- PS-10 provenance badge; NULL = born-digital
    CONSTRAINT documents_source_unique UNIQUE (source_type, source_id),
    CONSTRAINT documents_dual_approval CHECK (
        ingestion_status <> 'approved'
        OR (approved_by IS NOT NULL
            AND second_approved_by IS NOT NULL
            AND approved_by <> second_approved_by)
    )
);
CREATE INDEX IF NOT EXISTS documents_status_idx  ON documents (ingestion_status);
CREATE INDEX IF NOT EXISTS documents_hash_idx    ON documents (content_hash);

-- ---------------------------------------------------------------------------
-- chunks: one row per chunk — the retrieval unit. Typed metadata columns only
-- (no jsonb blob) so relational filtering runs before the vector index.
-- embedding: 1024-dim — fits both intfloat/multilingual-e5-large and
-- BAAI/bge-m3, so the bake-off (§7) does not require a schema change.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id          UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_type          ingestion_source_type NOT NULL,   -- denormalized for filter pushdown
    chunk_index          INT NOT NULL,                     -- position within document
    chunk_text           TEXT NOT NULL,                    -- REDACTED for nkp_case
    span_sha256          TEXT NOT NULL,                    -- hash of normalized chunk_text (§7.7 span verification)
    embedding            vector(1024),
    chunk_type           TEXT NOT NULL,                    -- section_label
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- NKP case metadata (NULL for act/regulation rows)
    case_id              TEXT,
    case_type            TEXT,
    court                TEXT,
    bench_type           TEXT,
    decision_date_ad     DATE,
    year_bs              INT,
    section_type         case_section_type,
    is_landmark          BOOLEAN,
    parties_redacted     TEXT,
    cited_statutes       TEXT[],
    headnotes            TEXT,

    -- law/regulation metadata (NULL for nkp_case rows)
    work_id              UUID REFERENCES work(id),
    act_name             TEXT,
    english_name         TEXT,
    document_type        TEXT,
    section_number       TEXT,
    section_title        TEXT,
    chapter_number       TEXT,
    level                law_level,
    parent_section       TEXT,
    effective_date_ad    DATE,                           -- from approved commence lifecycle_effect; NULL-pending (PS-2)
    co_retrieve_parent_id UUID REFERENCES chunks(id),    -- PS-16: proviso/subsection → operative clause

    -- shared LLM-extracted metadata
    keywords             TEXT[],
    relevant_questions   TEXT[],

    CONSTRAINT chunks_doc_index_unique UNIQUE (document_id, chunk_index),
    CONSTRAINT chunks_case_shape CHECK (
        source_type <> 'nkp_case'
        OR (case_id IS NOT NULL AND section_type IS NOT NULL)
    ),
    CONSTRAINT chunks_law_shape CHECK (
        source_type = 'nkp_case'
        OR (work_id IS NOT NULL AND section_number IS NOT NULL AND level IS NOT NULL)
    )
);

-- HNSW vector index (pgvector). m=16 / ef_construction=64 are the pgvector
-- documented defaults; ef_search is set at query time, not here.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Relational pre-filters (eligibility gate runs before kNN, §2 invariant 2)
CREATE INDEX IF NOT EXISTS chunks_case_filters_idx
    ON chunks (source_type, case_type, decision_date_ad, court)
    WHERE source_type = 'nkp_case';
CREATE INDEX IF NOT EXISTS chunks_work_idx      ON chunks (work_id) WHERE work_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_section_idx   ON chunks (work_id, section_number) WHERE work_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_parent_idx    ON chunks (co_retrieve_parent_id) WHERE co_retrieve_parent_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- pii_vault: unredacted NKP content (PS-14).
-- ACCESS CONTROL: readable ONLY by the `pii_vault_reader` role. The
-- application role used by the API/retrieval path must NEVER be granted this
-- role. Grants below are deliberate; do not add GRANTs for app roles here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pii_vault (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE RESTRICT,
    appellant_raw  TEXT,
    respondent_raw TEXT,
    full_text_raw  TEXT NOT NULL,
    stored_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pii_vault_reader') THEN
        CREATE ROLE pii_vault_reader NOLOGIN;
    END IF;
END $$;
REVOKE ALL ON pii_vault FROM PUBLIC;
GRANT SELECT ON pii_vault TO pii_vault_reader;
```

**Idempotency key mechanics:** `content_hash = sha256(NFC + digit-fold + whitespace-canonicalized raw content)`. On re-ingestion of an existing `(source_type, source_id)`: hash equal → skip entirely (no LLM calls, no embeddings); hash different → the document is an *amended source*, which enters the human-gated lifecycle path (new `documents.valid_time` range via an update that closes the old range, `ingestion_status` reset to `pending`) — never an in-place overwrite of approved state.

---

## 6. Pipeline stages

```
                         INGESTION PIPELINE (offline, human-gated)
                         =========================================

  laws.jsonl / nkp_cases.jsonl / regulations.json(+PDFs)
        │
        ▼
 ┌─────────────┐   in: raw JSONL/PDF          fail: malformed record → quarantine file,
 │ 1. LOAD     │   out: documents row (raw,   retry: none — fix source, re-run.
 │             │        status=pending)       parallel: across documents.
 └─────────────┘        content_hash check ──► same hash? SKIP (idempotent).
        │                                      diff hash? → lifecycle path, status=pending.
        ▼
 ┌─────────────┐   in: documents row          fail: NFC/digit-fold anomalies, empty
 │ 2. VALIDATE │   out: normalized text +     text, OCR score < floor → status=rejected
 │             │        quality flags         candidate w/ human review.
 └─────────────┘                              parallel: across documents.
        │
        ▼  (nkp_case only)
 ┌─────────────┐   in: normalized text,       fail: post-redaction token assertion fails
 │ 3. REDACT   │       appellant/respondent   → quarantine, status stays pending.
 │    _PII     │   out: redacted text →       retry: deterministic pass always re-runnable;
 │             │       documents.raw_content, haiku pass retried w/ backoff (rate limit).
 │             │       originals → pii_vault  parallel: across documents.
 └─────────────┘
        │
        ▼
 ┌─────────────┐   in: redacted text          fail: no anchors matched → single 'body'
 │ 4. CHUNK    │   out: chunk records w/      chunk (loud, logged) — never silent.
 │ (structure) │       section metadata,      parallel: across documents; pure CPU,
 │             │       co_retrieve links      no external calls, no retry needed.
 └─────────────┘
        │
        ▼
 ┌─────────────┐   in: chunk list per doc     fail: haiku 429/5xx → exponential backoff,
 │ 5. EXTRACT  │   out: keywords, questions,  max 5 retries, then chunk ships with NULL
 │  _METADATA  │       cited_statutes,        LLM columns (flagged). Batch API preferred.
 │  (haiku)    │       headnotes cleanup      parallel: across documents (rate-limited).
 └─────────────┘   ★ RATE-LIMIT CHOKEPOINT #1 (Anthropic API)
        │
        ▼
 ┌─────────────┐   in: chunk_text (amend      fail: embedder OOM/timeout → batch halve +
 │ 6. EMBED    │       tags stripped)         retry ×3. Self-hosted bge-m3 → throughput
 │             │   out: vector(1024)          bound by GPU; batch size 32.
 └─────────────┘   ★ CHOKEPOINT #2 (embedding throughput)   parallel: across documents
        │                                                    AND across chunks within one doc.
        ▼
 ┌─────────────┐   in: chunks + embeddings    fail: DB error → transaction rollback,
 │ 7. UPSERT   │   out: chunks rows           whole-document retry (idempotent via
 │             │       (status still pending) UNIQUE(document_id, chunk_index)).
 └─────────────┘                              parallel: across documents (advisory lock
        │                                      per document_id).
        ▼
 ╔═════════════╗
 ║ 8. DUAL     ║   PS-2 GATE: pipeline PAUSES here. Nothing above this line is
 ║  APPROVAL   ║   searchable. Gatekeeper + second approver review the document
 ║  (PS-2)     ║   (and any lifecycle_effect proposals); both names recorded on
 ╚═════════════╝   documents; CHECK constraint rejects single/self approval.
        │ approved
        ▼
  searchable (retrieval filters on ingestion_status='approved') +
  re-embed only affected components on later amendments (§5 Side A)
```

**Answers to the brief's questions:**

- **Parallelizable across documents**: every stage (1–7); documents are independent units.
- **Parallelizable within a document**: stage 6 (embed) only; chunking and metadata extraction are sequential per document (extraction needs the full chunk list for one batched call).
- **Rate-limit chokepoints**: stage 5 (haiku — use Message Batches API; ~1,022 cases × 2 calls + 677 acts × 1 call ≈ 3k requests) and stage 6 (embedding throughput — self-hosted bge-m3, batch 32, ~50–80k chunks total is hours on a single GPU).
- **Idempotency**: `content_hash` compared at stage 1, before any paid work; equal hash = hard skip. `UNIQUE(source_type, source_id)` is the database backstop.
- **PS-2 pause point**: between stage 7 and searchability — approval flips `ingestion_status` and is the only path to `approved`.

---

## 7. Open questions for Prakash

1. **BM25 / text-search (decision required before implementation — NOT silently resolved).** pgvector gives only kNN; §8 requires "BM25 Nepali":
   - **Option A — `tsvector` + GIN**: zero new infra, but PostgreSQL has **no Nepali text-search configuration** — no stemmer, no stopword list; we'd be on the `simple` config, which lowercases and splits on non-alphanumerics only. Devanagari is space-delimited so tokenization mostly works, but zero morphological normalization (e.g. गरेको/गर्ने/गरी never match) makes this the weakest recall option. Deterministic and cheap.
   - **Option B — OpenSearch + pgvector (two systems)**: best-in-class BM25, but still no mature Nepali analyzer (ICU tokenizer only), and every lifecycle write must now propagate to two derivatives with consistency drift between them — the system design's degraded ladder (§9) already prices in OpenSearch as a separate failure domain. Highest operational cost.
   - **Option C — `pg_search` (ParadeDB)**: BM25 inside PostgreSQL, same transaction boundary as the authority store, one system to reason about; Tantivy tokenizers handle Devanagari as unicode words but likewise offer **no Nepali stemmer**. Extension availability on the target Postgres host must be confirmed.
   - **My recommendation: Option C (pg_search)**, accepting that *all three* options lack Nepali stemming — so morphology is better recovered at query time (variant expansion in the understanding layer, §8) than by the index. If ParadeDB is unavailable on the host, fall back to Option A and put the effort into query-side variants instead of standing up OpenSearch for BM25 alone.
2. **Embedding model**: column is `vector(1024)`, which fits both `intfloat/multilingual-e5-large` and `BAAI/bge-m3`. **Recommend bge-m3** (stronger multilingual retrieval benchmarks, 8,192-token context, supports long NKP opinion chunks) with a bake-off on the golden eval slice before the full embed run — re-embedding ~60k chunks is the expensive operation to get right once. (`paraphrase-multilingual-mpnet-base-v2` is 768-dim and older; dropped.)
3. **Regulations have no text** (§1.3): confirm the fetch-and-convert path (Azure DI quota, PDF availability at the `regulations.json` URLs) and whether regulation ingestion is in PE-A scope or a follow-up.
4. **`is_landmark`**: my starting heuristic (bench type + headnote count) is crude. Real landmark status should derive from approved `precedent_relation` data once Phase D populates it; until then the column is a provisional flag — confirm that's acceptable.
5. **PS-2 (commencement) note**: the acts corpus itself contains delayed-commencement clauses (§1.2). `effective_date_ad` on chunks is therefore a **denormalized cache of the authority store's approved commence effect**, refreshed on lifecycle writes — not parsed text. Confirm the refresh trigger belongs to the lifecycle-write path, not this pipeline.

### PS-conflict self-check

No conflicts found. The design defers to: PS-2 (dual approval + NULL-pending effective dates), PS-3 (consolidated `laws.jsonl` content treated as derived, `<amend>` provenance preserved), PS-14 (pii_vault isolation + role restriction), PS-16 (`co_retrieve_parent_id` + mandatory parent fetch in context assembly), PS-5 (BS→AD only via `bs_ad_calendar`). One watch item: `chunks` is a derivative containing `effective_date_ad`; if that cache ever disagrees with `lifecycle_effect`, the authority store wins and the gate revalidates — the cache must never be trusted for validation (§2 invariant 1).
