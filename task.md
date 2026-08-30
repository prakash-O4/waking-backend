# AGENT-10 — Enabling-power links

## Objective

Subordinate नियमावली documents have no machine-readable link to the parent ऐन
section that authorises them. This means retrieval cannot automatically surface
the enabling provision when a user asks about a regulation's scope or validity —
and it means repealed enabling sections go undetected (PS-4 / §7.5).

This task:
1. Fixes a blocking migration gap from AGENT-9 (tariff `law_level` enum values).
2. Adds a `work_relations` table to store enabling-power links.
3. Extracts enabling clauses from नियमावली preambles during ingestion and writes
   `work_relations` rows.
4. Provides a post-processing script for the 6 नियमावली already in the DB.
5. Wires retrieval so नियमावली chunks co-retrieve their enabling provision.

## Assigned branch

`agent/enabling-power-links` (base: `dev` at current HEAD)

## Acceptance criteria

1. Migration `008_work_relations.sql` applies cleanly. `law_level` enum includes
   `tariff_heading`, `tariff_row`, `tariff_note` (unblocks भन्सार महसुल ऐन ingest).
2. `work_relations` is populated for every नियमावली/नियमहरू document at ingest time:
   - Resolved link: `enabling_work_id` + `enabling_section_number` set correctly.
   - Parent absent from corpus: `resolution_status='parent_not_in_corpus'`, `enabling_work_id=NULL`.
   - No regex match: `resolution_status='no_enabling_clause'`, explicit row (not silent).
3. Post-processing script (`scripts/backfill_enabling_links.py`) correctly fills
   `work_relations` for the 6 already-ingested नियमावली from DB `raw_content`.
4. Retrieval: when a नियमावली chunk appears in results, the enabling section chunk
   from the parent ऐन is co-retrieved and labelled `[CO-REF]` in context assembly.
   Enabling chunk must pass the eligibility gate — not bypass it.
5. `make test` green, `make lint` green, `make eval-gates` green.

## Files to create / modify

| Action | File |
|---|---|
| **CREATE** | `migrations/008_work_relations.sql` |
| **CREATE** | `app/ingestion/enabling_extractor.py` |
| **CREATE** | `scripts/backfill_enabling_links.py` |
| **CREATE** | `tests/test_enabling_extractor.py` |
| **CREATE** | `tests/test_enabling_retrieval.py` |
| **MODIFY** | `app/ingestion/pipeline.py` — call extractor after CHUNK stage |
| **MODIFY** | `app/retrieval/query_graph.py` — add `enabling_power_resolver_node` |

Do not touch: `laws_chunker.py`, `tariff_chunker.py`, `nkp_chunker.py`,
`pgvector_indexer.py`, `metadata_enricher.py`, `eligibility_gate.py`,
`postgres_retriever.py`, `authority/parser.py`, `authority/writer.py`.

## Step 0 — Fix law_level enum (blocking AGENT-9 ingest)

In `migrations/008_work_relations.sql`, add at the top before the CREATE TABLE:

```sql
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_heading';
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_row';
ALTER TYPE law_level ADD VALUE IF NOT EXISTS 'tariff_note';
```

These `ADD VALUE IF NOT EXISTS` statements are idempotent and safe to run on a
DB that already has these values (future-proof).

## Step 1 — `work_relations` table schema

```sql
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
```

## Step 2 — `app/ingestion/enabling_extractor.py`

New module. Put all extraction logic here so both the pipeline and the backfill
script share a single helper (no duplication).

### Regex — two variants

```python
import re, unicodedata

# Standard: "ऐन, २०७४ को दफा ४४ ले दिएको अधिकार"
_ENABLING_STRICT_RE = re.compile(
    r"([^\n।]{5,100})\s+को\s+(दफा|धारा)\s+([^\s,।]{1,20})"
    r"\s+ले\s+दिएको\s+अधिकार"
)

# उपदफा variant: "ऐन को दफा ४४ को उपदफा (२) ले दिएको अधिकार"
_ENABLING_UPADAFA_RE = re.compile(
    r"([^\n।]{5,100})\s+को\s+(दफा|धारा)\s+([^\s,।]{1,20})"
    r"\s+को\s+उपदफा\s+\(([^\)]{1,10})\)\s+ले\s+दिएको\s+अधिकार"
)
```

### Amend-markup strip

Apply before regex. Documents may have `<amend>...</amend>` or `</amend>` tags
in raw_content preamble — these can produce false positive matches.

```python
_AMEND_TAG_RE = re.compile(r"</?amend[^>]*>", re.IGNORECASE)

def _strip_amend_markup(text: str) -> str:
    return _AMEND_TAG_RE.sub("", text)
```

### Title normalization for work resolution

`work.title_ne` stores years without commas (e.g. `'लोक सेवा आयोग ऐन २०७९'`)
but the corpus preamble captures `'लोक सेवा आयोग ऐन, २०७९'`. Normalize before
matching:

```python
def _normalize_title(text: str) -> str:
    text = re.sub(r",\s*", " ", text)          # strip commas
    text = unicodedata.normalize("NFC", text)
    return text.strip()
```

### Section-number normalization

Strip the `दफा`/`धारा` prefix and normalize Devanagari digits to ASCII for DB
storage (retrieval JOIN uses ASCII-normalized `section_number`):

```python
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def _normalize_section_num(s: str) -> str:
    return s.strip().translate(_DEVA_DIGITS)
```

### Main extraction function

```python
def extract_enabling_clause(
    content: str,
    work_id: str,           # subordinate work UUID (already in DB)
    conn,                   # psycopg2 connection
    source_id: str = "",    # for logging
) -> None:
    """
    Extract enabling clause from first 1500 chars of content and write to
    work_relations. Always writes exactly one row (resolved, unresolved, or
    no_enabling_clause sentinel). Idempotent via ON CONFLICT DO NOTHING.
    """
    preamble = _strip_amend_markup(content[:1500])
    row = _parse_enabling_clause(preamble)

    if row is None:
        _insert_relation(conn, work_id, None, None, None, None,
                         raw_clause_text="", status="no_enabling_clause")
        return

    provision_type, section_num_raw, subsection_num_raw, act_ref_raw = row
    section_num = _normalize_section_num(section_num_raw)
    subsection_num = _normalize_section_num(subsection_num_raw) if subsection_num_raw else None
    act_ref_norm = _normalize_title(act_ref_raw)

    enabling_work_id = _resolve_work(conn, act_ref_norm)
    status = "auto_extracted" if enabling_work_id else "parent_not_in_corpus"

    _insert_relation(
        conn, work_id, enabling_work_id, provision_type,
        section_num, subsection_num,
        raw_clause_text=f"{act_ref_raw} को {provision_type} {section_num_raw}",
        status=status,
    )
```

### Work resolution

```python
def _resolve_work(conn, normalized_title: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM work WHERE title_ne = %s LIMIT 1",
            (normalized_title,)
        )
        row = cur.fetchone()
    return str(row[0]) if row else None
```

Do not use partial/ILIKE matching — false positives on short titles (e.g. "कर
ऐन") would incorrectly link dozens of regulations to the wrong parent. Exact
match after comma-normalization is the safest deterministic approach.

### `_insert_relation` helper

Use `ON CONFLICT DO NOTHING` on both partial indexes so the function is
idempotent — calling it twice for the same document does not create duplicate rows.

## Step 3 — Pipeline integration

In `app/ingestion/pipeline.py`, after the CHUNK stage and before EMBED, add:

```python
from app.ingestion.enabling_extractor import extract_enabling_clause

_NIYAM_SUFFIXES = ("नियमावली", "नियमहरू", "नियम")

# inside ingest_law(), after chunks are produced:
doc_name = record.get("name", "")
if any(doc_name.endswith(s) for s in _NIYAM_SUFFIXES):
    extract_enabling_clause(
        content=content,
        work_id=work_id,    # the UUID returned by upsert_work
        conn=self._conn,
        source_id=record.get("source_id", ""),
    )
```

`work_id` is the UUID from the existing `upsert_work` call earlier in the
pipeline. Do not add a new DB round-trip to look it up.

The enabling extraction happens before embedding so any failure (wrong title
normalization, missing parent) is visible in the same pipeline run log without
blocking the chunk ingest.

## Step 4 — Post-processing script

`scripts/backfill_enabling_links.py`:

- Query `documents` table for `source_type = 'law'` where the source_id ends
  with a नियम suffix (or use `raw_content` ILIKE on preamble keywords).
- For each, call `extract_enabling_clause(raw_content, work_id, conn)`.
- Print a summary: resolved / parent_not_in_corpus / no_enabling_clause counts.
- Idempotent: run multiple times safely.

Expected result: 6 rows inserted (the already-ingested नियमावली).

## Step 5 — Retrieval wiring

Add a new node `enabling_power_resolver_node` in `app/retrieval/query_graph.py`,
wired **after** `cross_ref_resolver_node` and **before** the reasoner node.

Do NOT extend `_resolve_cross_refs` — that function is intra-document. Enabling
power is cross-document.

```python
def enabling_power_resolver_node(state: AgentState, config: RunnableConfig):
    conn = config["configurable"]["conn"]
    as_of = state["session_as_of"]
    hits = state.get("all_hits", [])
    additional = []

    for h in hits[:5]:           # only top-5 to bound latency
        if h.get("co_retrieved"):
            continue
        enabling = _fetch_enabling_chunk(conn, h, as_of)
        if enabling:
            additional.append({**enabling, "co_retrieved": True})

    if not additional:
        return {}
    return {"all_hits": hits + additional}
```

`_fetch_enabling_chunk` logic:

```python
def _fetch_enabling_chunk(conn, hit, as_of):
    # 1. look up work_relations for this chunk's work_id
    # 2. if enabling_work_id is NULL → return None
    # 3. fetch chunk WHERE work_id = enabling_work_id
    #    AND section_number = wr.enabling_section_number
    #    AND chunk_type = 'dafa {section_number}'
    #    LIMIT 1
    # 4. check eligibility gate: chunk must be in eligible_chunk_ids(conn, as_of)
    #    if not eligible, return None (do not bypass PS-6)
    # 5. return chunk dict or None
```

The eligibility gate check is non-negotiable (PS-6/PS-4): a repealed enabling
section must NOT be co-retrieved as if it were current law.

## Relevant design refs

- `system-design.md` §7.5, PS-4, PS-6, PS-16
- `app/retrieval/eligibility_gate.py` — use existing `eligible_chunk_ids()`
- `app/retrieval/query_graph.py` — follow the existing node pattern
- `app/ingestion/laws_chunker.py` — `section_number` normalization conventions
- `migrations/001_bitemporal_schema.sql` — `work` table columns (no source_id;
  resolution goes via `work.title_ne`)
- `migrations/005_ingestion_pipeline.sql` — `law_level` enum, `chunks` table

## PS requirements in scope

- **PS-4** — enabling-power link must be stored and resolvable from DB. Unresolved
  cases get explicit `resolution_status` rows — never silent nulls.
- **PS-6** — enabling chunk must pass `eligible_chunk_ids` before co-retrieval.
  A repealed enabling section must not appear in results.
- **PS-16** — co-retrieved enabling chunk follows the same labelling path as
  स्पष्टीकरण co-retrieval; `co_retrieved=True` flag preserved through context assembly.
- **PS-2** — `work_relations` rows are written automatically (not human-gated)
  and carry `resolution_status='auto_extracted'`. This is explicitly acceptable
  for Phase 0: they are derivative metadata, not legal-state changes (those go
  through `lifecycle_effect` with the dual-approval path). Do not add approval
  gating to this table at this stage.

## Zero-tolerance gates — do not touch

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`
- `overruled-as-good-law = 0`

The enabling-power resolver must call `eligible_chunk_ids` — it must never return
a repealed chunk regardless of what `work_relations` contains.

## Required checks

```
make test
make lint
make eval-gates
```

## Tests to write

### `tests/test_enabling_extractor.py`

1. `test_enabling_regex_standard` — standard `दफा X ले दिएको अधिकार` → captures
   act ref, provision type `'dafa'`, section number correctly
2. `test_enabling_regex_upadafa_variant` — `दफा X को उपदफा (Y) ले दिएको अधिकार`
   → broad regex fires; `enabling_subsection_number` populated
3. `test_enabling_regex_amend_markup_no_false_positive` — preamble with
   `</amend>` tag → after strip, regex does not fire on tag text
4. `test_title_normalization_comma` — `'लोक सेवा आयोग ऐन, २०७९'` →
   normalized to `'लोक सेवा आयोग ऐन २०७९'` (comma stripped)
5. `test_section_normalization_devanagari` — `'४४'` → `'44'` (Devanagari to ASCII)
6. `test_extract_inserts_resolved_row` — mock `_resolve_work` returns a UUID →
   row with `resolution_status='auto_extracted'`, correct IDs
7. `test_extract_inserts_parent_not_in_corpus` — `_resolve_work` returns None →
   row with `resolution_status='parent_not_in_corpus'`, `enabling_work_id=NULL`
8. `test_extract_inserts_no_clause_sentinel` — no regex match →
   row with `resolution_status='no_enabling_clause'`
9. `test_extract_idempotent` — call twice for same work_id → exactly one row
   (ON CONFLICT DO NOTHING)

### `tests/test_enabling_retrieval.py`

10. `test_enabling_chunk_passes_eligibility_gate` — enabling chunk not in
    `eligible_chunk_ids` → resolver returns None, chunk not added to hits
11. `test_enabling_resolver_coretrieves` — happy path: नियमावली chunk in hits,
    resolved enabling section in DB and eligible → co-retrieved chunk appended
    with `co_retrieved=True`
12. `test_enabling_resolver_null_link_skipped` — `enabling_work_id=NULL` in
    `work_relations` → resolver returns None, no co-retrieval attempted

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution of any kind.

## Return

Commit hash, changed files, checks run + results, and explicit answers to:
- How many rows did the backfill script insert for the 6 already-ingested नियमावली?
- Were any resolution_status='parent_not_in_corpus' among those 6?
- Did `make eval-gates` stay at zero on all three zero-tolerance counters?
- Any enabling clause variants in the actual corpus that the regex did not catch?
