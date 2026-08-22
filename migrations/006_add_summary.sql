-- migrations/006_add_summary.sql
-- Add document-level summary column (LLM-extracted; NULL until enriched).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT;
