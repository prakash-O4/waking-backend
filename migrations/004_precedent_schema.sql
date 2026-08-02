-- A Supreme Court case (नजिर)
CREATE TABLE IF NOT EXISTS precedent (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uri          TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    decided_date DATE,
    bench_size   INT NOT NULL CHECK (bench_size > 0),
    court        TEXT NOT NULL DEFAULT 'supreme_court',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A specific holding/proposition within a case.
-- Overruling is at holding level, never at case level.
CREATE TABLE IF NOT EXISTS precedent_holding (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    precedent_id  UUID NOT NULL REFERENCES precedent(id) ON DELETE CASCADE,
    holding_text  TEXT NOT NULL,
    source_span   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Temporal relation between a source case and a target holding.
-- Human-extracted, dual-approval gated (same pattern as lifecycle_effect).
CREATE TABLE IF NOT EXISTS precedent_relation (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_case_id    UUID NOT NULL REFERENCES precedent(id),
    target_holding_id UUID NOT NULL REFERENCES precedent_holding(id),
    relation_type     TEXT NOT NULL CHECK (relation_type IN (
                          'overrules','reverses','distinguishes','affirms','questions'
                      )),
    bench_strength    INT NOT NULL CHECK (bench_strength > 0),
    legal_valid_time  TSTZRANGE NOT NULL DEFAULT tstzrange(now(), 'infinity'),
    source_span       TEXT,
    approval_status   TEXT NOT NULL DEFAULT 'pending'
                          CHECK (approval_status IN ('pending','approved','rejected')),
    approved_by_1     UUID,
    approved_by_2     UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Holding-level good-law derivation.
-- Returns FALSE if any approved overrule/reverse from a competent-or-larger bench
-- covers the query date. TRUE otherwise (including per-incuriam smaller-bench rulings).
CREATE OR REPLACE FUNCTION is_good_law(p_holding_id UUID, p_as_of DATE)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN NOT EXISTS (
        SELECT 1
        FROM precedent_relation pr
        JOIN precedent source_case  ON source_case.id = pr.source_case_id
        JOIN precedent_holding ph   ON ph.id = pr.target_holding_id
        JOIN precedent target_case  ON target_case.id = ph.precedent_id
        WHERE pr.target_holding_id = p_holding_id
          AND pr.relation_type IN ('overrules', 'reverses')
          AND source_case.bench_size >= target_case.bench_size
          AND pr.approval_status = 'approved'
          AND pr.legal_valid_time @> p_as_of::TIMESTAMPTZ
    );
END;
$$ LANGUAGE plpgsql;
