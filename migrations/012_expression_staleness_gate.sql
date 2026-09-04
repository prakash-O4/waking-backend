CREATE OR REPLACE FUNCTION is_expression_current(
    p_component_uri    TEXT,
    p_chunk_created_at  TIMESTAMPTZ,
    p_as_of             DATE
) RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM lifecycle_effect
        WHERE component_uri = p_component_uri
          AND effect_type = 'amend'
          AND approval_status = 'approved'
          AND lower(legal_valid_time) <= p_as_of::timestamptz
          AND lower(transaction_time) > p_chunk_created_at
    );
$$;
