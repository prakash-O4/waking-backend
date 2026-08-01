# Task P0-A — Phase 0 Infrastructure Skeleton

**Engineer:** Pi  
**Branch:** `phase-0/infra-skeleton`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Wire the complete infrastructure skeleton for Wakil-G so that every subsequent Phase 0 task has a stable surface to build on. No generation logic yet — skeleton only. The deliverable is a repo where `make setup` runs without error, `make test` runs (even if zero tests pass), and the bitemporal schema is migrated into the Supabase Postgres instance.

---

## Acceptance criteria

1. `make setup` installs all deps and runs the Supabase migration without error.
2. `make test` runs (pytest) — zero failures (zero tests is acceptable; the harness must exist).
3. `make lint` runs (ruff) — zero errors on existing code.
4. `make eval`, `make eval-gates`, `make stress` print a clear "no cases yet" message and exit 0.
5. Supabase Postgres has the five bitemporal tables created by the migration (verified by a `make test` fixture that connects and lists tables).
6. OpenSearch is reachable via `make setup` check (Docker Compose up + health ping); the index mapping is created.
7. Python models for all five authority entities exist and pass `mypy --strict` (or ruff type-check).
8. `requirements.txt` is updated; no unused deps added.

---

## Allowed scope

- Create: `Makefile`, `migrations/`, `app/authority/`, `app/search/`, `tests/`, `docker-compose.yml`
- Modify: `requirements.txt`
- Do NOT modify: `app/main.py`, `app/retrieval/`, `app/ingestion/` — those are Phase 0-B

---

## Forbidden changes

- Do not touch `app/main.py` or any existing retrieval/ingestion files.
- Do not add any LLM calls, embeddings, or generation logic.
- Do not add LangGraph yet (Phase B).
- Do not add Pinecone anything — we are replacing it with OpenSearch.

---

## Tech stack for this task

- **Postgres:** Supabase (connection via `SUPABASE_DB_URL` env var — direct Postgres connection string, not the REST client). Use `psycopg2-binary` for migrations.
- **OpenSearch:** Local via Docker Compose (`opensearchproject/opensearch:2.x`). Client: `opensearch-py`.
- **Linting:** `ruff` (add to dev deps).
- **Tests:** `pytest` + `pytest-asyncio`.
- **Type checking:** `mypy` or ruff's type rules.

---

## Bitemporal schema — exact tables to create

Migration file: `migrations/001_bitemporal_schema.sql`

### work
```sql
CREATE TABLE work (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uri         TEXT NOT NULL UNIQUE,           -- e.g. /np/act/2063/sarbajanik
    work_type   TEXT NOT NULL,                  -- Constitution / Act / Rule / Directive / Notification
    title_ne    TEXT NOT NULL,
    title_en    TEXT,
    jurisdiction TEXT NOT NULL DEFAULT 'NP',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### component
```sql
CREATE TABLE component (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id         UUID NOT NULL REFERENCES work(id),
    uri             TEXT NOT NULL UNIQUE,        -- /np/act/2063/sarbajanik/dafa/1
    component_type  TEXT NOT NULL,               -- dafa / updafa / parichheda / proviso / spastikaran / anusuchi
    number          TEXT,
    parent_uri      TEXT,                        -- NULL for top-level
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### source_publication
```sql
CREATE TABLE source_publication (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_id         UUID NOT NULL REFERENCES work(id),
    kind            TEXT NOT NULL CHECK (kind IN (
                        'official_original',
                        'amending_instrument',
                        'verified_internal_consolidation',
                        'official_copy_unverified',
                        'derived_verified'
                    )),
    source_url      TEXT,
    sha256          TEXT NOT NULL,               -- SHA-256 of raw source bytes
    ocr_confidence  FLOAT,                       -- NULL if not OCR'd
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### lifecycle_effect
```sql
CREATE TABLE lifecycle_effect (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uri           TEXT NOT NULL,
    effect_type             TEXT NOT NULL CHECK (effect_type IN (
                                'amend','repeal','commence','expiry',
                                'suspend','correct','declared_invalid'
                            )),
    legal_valid_time        TSTZRANGE NOT NULL,   -- when this effect is law
    transaction_time        TSTZRANGE NOT NULL,   -- when the system recorded it
    effective_date          DATE,                 -- NULL if commencement pending
    commencement_dependency TEXT,                 -- 'gazette_notification' or NULL
    replacement_text        TEXT,
    source_pub_id           UUID REFERENCES source_publication(id),
    approval_status         TEXT NOT NULL DEFAULT 'pending'
                                CHECK (approval_status IN ('pending','approved','rejected')),
    approved_by_1           UUID,                 -- first approver user_id
    approved_by_2           UUID,                 -- second approver user_id
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT no_overlap EXCLUDE USING GIST (
        component_uri WITH =,
        legal_valid_time WITH &&
    ) WHERE (approval_status = 'approved')
);
```

### expression
```sql
CREATE TABLE expression (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uri   TEXT NOT NULL,
    as_of           DATE NOT NULL,
    text_ne         TEXT NOT NULL,               -- derived text (non-authoritative)
    text_hash       TEXT NOT NULL,               -- SHA-256 of text_ne (for span verification)
    is_derived      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## Eligibility gate — SQL function

Create in migration:

```sql
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
        -- Excludes: pending, rejected, not-yet-effective, repealed-after
    ) AND NOT EXISTS (
        SELECT 1 FROM lifecycle_effect
        WHERE component_uri = p_component_uri
          AND effect_type   IN ('repeal','expiry','declared_invalid')
          AND approval_status = 'approved'
          AND lower(legal_valid_time) <= p_as_of::timestamptz
    );
$$;
```

---

## Python models — `app/authority/models.py`

Pydantic v2 models mirroring every table above. Include:
- `WorkType` enum
- `ComponentType` enum  
- `SourceKind` enum
- `EffectType` enum
- `ApprovalStatus` enum
- `Work`, `Component`, `SourcePublication`, `LifecycleEffect`, `Expression` models

No ORM. Plain Pydantic for validation; DB access is raw SQL via psycopg2.

---

## OpenSearch — `docker-compose.yml` + `app/search/client.py`

Docker Compose:
```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:2.13.0
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
    ports:
      - "9200:9200"
```

`app/search/client.py`:
- `get_client()` — returns `OpenSearch` instance from `OPENSEARCH_URL` env (default `http://localhost:9200`)
- `ensure_index(index_name)` — creates index with mapping below if not exists
- Index mapping: `text_ne` field with `nori` tokenizer fallback to `standard`, plus `dense_vector` field (dim=1536) for kNN

---

## Makefile targets

```makefile
setup:      ## Install deps, run migration, start OpenSearch, create index
test:       ## pytest tests/
lint:       ## ruff check . && ruff format --check .
eval:       ## python -m app.eval.runner (print "no eval cases yet" if empty)
eval-gates: ## python -m app.eval.gates (zero-tolerance gate check)
stress:     ## python -m app.stress.runner (print "no stress cases yet" if empty)
```

`eval`, `eval-gates`, and `stress` targets must exit 0 even when no cases exist. They must NOT import any LLM or embedding code at import time (stub-safe).

---

## Required checks before committing

```
make lint      # zero errors
make test      # zero failures
make setup     # runs clean
```

No `make eval-gates` requirement yet — stub is sufficient.

---

## Zero-tolerance gates guarded in this task

None (schema only, no ingestion or retrieval). No gate can fire without data.

---

## Commit authorship

Every commit MUST be authored as:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
**Never** add `Co-Authored-By: Claude`, "Generated with Claude", or any AI attribution. This is non-negotiable.

---

## Return to Claude (via Prakash)

When done, return:
- Commit hash
- Changed/created files list
- Output of `make lint`, `make test`, `make setup`
- Any assumptions made
- Any remaining risks
