-- Document LLM-derived metadata provenance for databases that already applied 005/006.
COMMENT ON COLUMN chunks.keywords IS
    'LLM-derived during ingestion, never human-reviewed, not authoritative, and must never be rendered as or substituted for statutory text or a citation.';
COMMENT ON COLUMN chunks.relevant_questions IS
    'LLM-derived during ingestion, never human-reviewed, not authoritative, and must never be rendered as or substituted for statutory text or a citation.';
COMMENT ON COLUMN documents.summary IS
    'LLM-derived during ingestion, never human-reviewed, not authoritative, and must never be rendered as or substituted for statutory text or a citation.';
