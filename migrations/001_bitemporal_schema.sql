CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE IF NOT EXISTS work (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uri          TEXT NOT NULL UNIQUE,
    work_type    TEXT NOT NULL,
    title_ne     TEXT NOT NULL,
    title_en     TEXT,
    jurisdiction TEXT NOT NULL DEFAULT 'NP',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS component (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id        UUID NOT NULL REFERENCES work(id),
    uri            TEXT NOT NULL UNIQUE,
    component_type TEXT NOT NULL,
    number         TEXT,
    parent_uri     TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_publication (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id        UUID NOT NULL REFERENCES work(id),
    kind           TEXT NOT NULL CHECK (kind IN (
                       'official_original',
                       'amending_instrument',
                       'verified_internal_consolidation',
                       'official_copy_unverified',
                       'derived_verified'
                   )),
    source_url     TEXT,
    sha256         TEXT NOT NULL,
    ocr_confidence FLOAT,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lifecycle_effect (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uri           TEXT NOT NULL,
    effect_type             TEXT NOT NULL CHECK (effect_type IN (
                                'amend','repeal','commence','expiry',
                                'suspend','correct','declared_invalid'
                            )),
    legal_valid_time        TSTZRANGE NOT NULL,
    transaction_time        TSTZRANGE NOT NULL,
    effective_date          DATE,
    commencement_dependency TEXT,
    replacement_text        TEXT,
    source_pub_id           UUID REFERENCES source_publication(id),
    approval_status         TEXT NOT NULL DEFAULT 'pending'
                                CHECK (approval_status IN ('pending','approved','rejected')),
    approved_by_1           UUID,
    approved_by_2           UUID,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT no_overlap EXCLUDE USING GIST (
        component_uri WITH =,
        legal_valid_time WITH &&
    ) WHERE (approval_status = 'approved')
);

CREATE TABLE IF NOT EXISTS expression (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uri TEXT NOT NULL,
    as_of         DATE NOT NULL,
    text_ne       TEXT NOT NULL,
    text_hash     TEXT NOT NULL,
    is_derived    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION is_eligible(
    p_component_uri TEXT,
    p_as_of         DATE
) RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT EXISTS (
        SELECT 1 FROM lifecycle_effect
        WHERE component_uri = p_component_uri
          AND effect_type   = 'commence'
          AND approval_status = 'approved'
          AND legal_valid_time @> p_as_of::timestamptz
          AND (commencement_dependency IS NULL)
    ) AND NOT EXISTS (
        SELECT 1 FROM lifecycle_effect
        WHERE component_uri = p_component_uri
          AND effect_type   IN ('repeal','expiry','declared_invalid')
          AND approval_status = 'approved'
          AND lower(legal_valid_time) <= p_as_of::timestamptz
    );
$$;
