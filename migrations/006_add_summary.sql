-- migrations/006_add_summary.sql
-- Add document-level summary column (LLM-extracted; NULL until enriched).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT;
COMMENT ON COLUMN documents.summary IS
    'LLM-derived during ingestion, never human-reviewed, not authoritative, and must never be rendered as or substituted for statutory text or a citation.';
