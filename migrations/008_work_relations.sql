-- migrations/008_work_relations.sql
-- Enabling-power links between subordinate regulations (नियमावली/नियमहरू)
-- and the parent ऐन provisions that authorise them (AGENT-10).
-- Also unblocks AGENT-9 tariff ingest by widening the law_level enum.

-- Step 0: AGENT-9 tariff levels must exist before any tariff document is ingested.
-- ALTER TYPE ADD VALUE IF NOT EXISTS is safe to re-run.
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_heading';
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_row';
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_note';

-- ---------------------------------------------------------------------------
-- work_relations: machine-readable links between works.
-- relation_type is currently limited to 'enabling_power'; reserved for future
-- relation kinds (e.g. amends, repeals) once the lifecycle pipeline supports them.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS work_relations (
    id                         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subordinate_work_id        UUID NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    relation_type              TEXT NOT NULL CHECK (relation_type = 'enabling_power'),
    enabling_work_id           UUID REFERENCES work(id),
    enabling_provision_type    TEXT CHECK (enabling_provision_type IN ('dafa', 'dhara')),
    enabling_section_number    TEXT,         -- normalized number only, no "दफा" prefix
    enabling_subsection_number TEXT,         -- populated for उपदफा variant; NULL otherwise
    raw_clause_text            TEXT NOT NULL, -- verbatim extracted clause (audit trail)
    resolution_status          TEXT NOT NULL DEFAULT 'auto_extracted'
                               CHECK (resolution_status IN (
                                   'auto_extracted',       -- extracted, not human-verified
                                   'human_verified',
                                   'parent_not_in_corpus', -- regex matched, parent act absent
                                   'no_enabling_clause',   -- no regex match; explicit sentinel
                                   'false_positive'        -- human-marked bad extraction
                               )),
    extracted_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_by                TEXT,
    valid_time                 TSTZRANGE NOT NULL DEFAULT tstzrange(now(), NULL)
);

-- Section-aware: allows multiple enabling provisions per नियमावली
CREATE UNIQUE INDEX IF NOT EXISTS work_relations_link_unique_idx
    ON work_relations (subordinate_work_id, enabling_work_id, enabling_section_number)
    WHERE enabling_work_id IS NOT NULL;

-- One sentinel row per subordinate when no clause found
CREATE UNIQUE INDEX IF NOT EXISTS work_relations_null_unique_idx
    ON work_relations (subordinate_work_id)
    WHERE enabling_work_id IS NULL AND resolution_status = 'no_enabling_clause';

CREATE INDEX IF NOT EXISTS work_relations_enabling_idx
    ON work_relations (enabling_work_id)
    WHERE enabling_work_id IS NOT NULL;
