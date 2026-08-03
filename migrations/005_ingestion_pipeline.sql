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
    -- Addition to the design-doc §5 DDL (task.md pipeline stage 3): set when
    -- post-redaction token verification fails; document stays 'pending' and is
    -- quarantined for human review, never silently passed.
    redaction_failed   BOOLEAN NOT NULL DEFAULT FALSE,
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
        OR (work_id IS NOT NULL AND level IS NOT NULL)
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

-- pg_search (ParadeDB): BM25 full-text index on chunks.
-- REQUIRES: ParadeDB extension installed on the PostgreSQL host.
-- If unavailable, comment this out and fall back to GIN tsvector (Option A).
CREATE EXTENSION IF NOT EXISTS pg_search;

-- BM25 index for Nepali text search. ParadeDB tokenizes on unicode word boundaries
-- (no Nepali stemmer — morphological variants handled at query time via variant expansion).
-- VERSION ASSUMPTION: the CALL paradedb.create_bm25(...) form below is correct for
-- ParadeDB ≥0.8. If the installed version differs, check
-- SELECT paradedb.schema_version() at runtime and adjust the syntax accordingly.
CALL paradedb.create_bm25(
    index_name => 'chunks_bm25',
    table_name => 'chunks',
    key_field  => 'id',
    text_fields => paradedb.field('chunk_text', tokenizer => paradedb.tokenizer('unicode'))
);
