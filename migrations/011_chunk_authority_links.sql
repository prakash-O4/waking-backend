ALTER TABLE chunks ADD COLUMN IF NOT EXISTS component_uri TEXT;

CREATE INDEX IF NOT EXISTS chunks_component_uri_idx
    ON chunks (component_uri) WHERE component_uri IS NOT NULL;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_pub_id UUID
    REFERENCES source_publication(id);
