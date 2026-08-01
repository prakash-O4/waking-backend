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
          AND effect_type   IN ('repeal','expiry','declared_invalid','suspend')
          AND approval_status = 'approved'
          AND lower(legal_valid_time) <= p_as_of::timestamptz
    );
$$;
