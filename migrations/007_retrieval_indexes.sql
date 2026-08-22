-- GIN index for tsvector lexical search on chunk_text
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_text_fts
    ON chunks USING GIN (to_tsvector('simple', chunk_text));
