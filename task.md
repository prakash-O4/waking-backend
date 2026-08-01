# Task P0-B — Phase 0 End-to-End: Full Corpus Ingestion + Dumb Baseline

**Engineer:** Pi  
**Branch:** `phase-0/end-to-end`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Ingest the full `laws.jsonl` corpus (677 laws) into the Supabase bitemporal authority store and OpenSearch, wire a dumb BM25 retrieval chain with eligibility gate and validation gate, and confirm `make eval-gates` is green. Do not touch `app/main.py`.

---

## Acceptance criteria

1. `scripts/ingest_laws.py` runs to completion and all 677 laws are registered in Supabase + OpenSearch (idempotent — safe to re-run).
2. `scripts/query.py "<question>" --as-of <YYYY-MM-DD>` returns an answer with citations rendered from canonical metadata (not from model output).
3. `make eval-gates` exits 0 and prints:
   ```
   repealed-as-current: 0
   not-yet-effective-as-current: 0
   ```
4. `make test` still green.
5. `make lint` still green.
6. The three P0-A gaps are fixed.

---

## Fix first — P0-A known gaps

**F1 — `app/search/client.py`:** Replace `nori_tokenizer` (Korean) with try-order:
```python
for tokenizer in ("icu_tokenizer", "standard"):
    try:
        client.indices.create(index=index_name, body=_mapping(tokenizer))
        return
    except RequestError:
        if client.indices.exists(index=index_name):
            return
raise RuntimeError("Could not create OpenSearch index")
```

**F2 — `app/authority/models.py`:** Add to `ComponentType`:
```python
BHAG = "bhag"      # भाग
KHANDA = "khanda"  # खण्ड
```

**F3 — Eligibility gate:** Add `suspend` to NOT EXISTS clause. Create `migrations/002_gate_suspend_fix.sql` (do NOT re-run 001) with just the updated `CREATE OR REPLACE FUNCTION is_eligible(...)`.

---

## Allowed scope

**Create:**
- `migrations/002_gate_suspend_fix.sql`
- `app/authority/writer.py`
- `app/authority/parser.py`
- `app/retrieval/eligibility_gate.py`
- `app/retrieval/dumb_retriever.py`
- `app/retrieval/validation_gate.py`
- `app/eval/__init__.py`
- `app/eval/gates.py`
- `scripts/ingest_laws.py`
- `scripts/query.py`
- `tests/test_eligibility_gate.py`
- `tests/test_validation_gate.py`
- `tests/test_parser.py`

**Modify:**
- `app/search/client.py` (F1)
- `app/authority/models.py` (F2)
- `scripts/migrate.py` (apply migration 002 as well)
- `Makefile` (wire `eval-gates`)

**Do NOT modify:**
- `app/main.py`
- `app/retrieval/advanced_retriever.py`, `query_processor.py`, `retrieval_orchestrator.py`
- `app/ingestion/` existing files
- `migrations/001_bitemporal_schema.sql`

---

## Forbidden

- No Pinecone imports in any new file.
- No Cohere reranker in new files.
- No LangGraph.
- Model must emit `claims + evidence_ids` only — no citations from the model.
- No path skips the eligibility gate.
- No path bypasses the validation gate.

---

## Corpus — `laws.jsonl`

Located at repo root. 677 JSONL records, one law per line.

**Fields:**
```json
{
  "url": "https://storage.googleapis.com/eksana_legal/Acts/...",
  "name": "भ्रष्टाचार_निवारण_ऐन_२०५९",
  "english_name": "Prevention of Corruption Act, 2002",
  "document_type": "act",
  "page": null,
  "content": "<full Nepali text>",
  "_id": "..."
}
```

**Content structure (672/677 laws):**
- BS enactment date near the top: `प्रमाणीकरण र प्रकाशित मिति\n२०५९।०३।०५`
- Section headers: `**दफा X.**`, `दफा X.`, `परिच्छेद-X`, `धारा X`
- Inline amendment markup: `<amend>...</amend>` — strip before storing in `text_ne`

---

## Parser — `app/authority/parser.py`

Parse each `laws.jsonl` record into structured data. This is the single place all corpus-specific logic lives.

```python
@dataclass
class ParsedLaw:
    uri: str                    # /np/act/2059/bhrastachar-nivarana
    work_type: WorkType
    title_ne: str               # cleaned name (underscores→spaces, URL-decoded)
    title_en: str | None
    enactment_ad: date | None   # converted from BS (see below)
    source_sha256: str          # SHA-256 of raw content bytes
    components: list[ParsedComponent]

@dataclass
class ParsedComponent:
    uri: str                    # /np/act/2059/bhrastachar-nivarana/dafa/1
    component_type: str         # dafa / parichheda / dhara / full (fallback)
    number: str | None
    text_ne: str                # <amend> tags stripped
    text_hash: str              # SHA-256 of text_ne.encode('utf-8')
```

### URI derivation

```python
def make_uri(name: str, doc_type: str) -> str:
    # name = "भ्रष्टाचार_निवारण_ऐन_२०५९"
    # 1. Extract year suffix: last token after final underscore that looks like
    #    a 4-digit Nepali number (२०५९ etc.)
    # 2. Convert Devanagari digits → ASCII digits
    # 3. Transliterate name for slug using slugify or simple replacement
    # 4. Return /np/{doc_type}/{bs_year}/{slug}
    # Example: /np/act/2059/bhrastachar-nivarana-ain
```

For the slug, replace underscores with hyphens, drop the year suffix, use `unidecode` or a simple Devanagari→Roman mapping (just for the URI — display uses original). Do not add `unidecode` as a dependency; use a minimal mapping or just use the `_id` field if it provides a stable slug.

Simple fallback: `uri = f"/np/{doc_type}/{bs_year}/{record['_id']}"` — use `_id` as the slug. It's stable and unique.

### BS → AD date conversion

**BS↔AD is Phase C.** For Phase 0, use a conservative approximation only:

```python
def bs_to_ad_approx(bs_year: int, bs_month: int) -> date:
    # Nepali calendar: BS year ≈ AD year + 56 or 57 depending on month.
    # BS months 1-3 (Baisakh-Ashadh) → BS year - 57 + 1 = AD year
    # BS months 4-12 → BS year - 56 = AD year (approximate)
    # This is intentionally rough — BS↔AD calendar module is Phase C.
    ad_year = bs_year - 57 if bs_month <= 3 else bs_year - 56
    return date(ad_year, 1, 1)  # Day/month approximated as Jan 1; use for valid_time only
```

Flag this with a `# PHASE-C-TODO: replace with canonical bs_ad_calendar lookup` comment.

If the date cannot be parsed (5/677 laws lack dates), set `enactment_ad = None` and skip lifecycle_effect creation for that law (log a warning).

### Section splitting

Split content at section headers using this pattern (in order of preference):

1. `**दफा \d+[\.।]**` — bold dafa (most common in this corpus)
2. `दफा \d+[\.।\s]` — plain dafa
3. `परिच्छेद[-–]\s*\d+` — chapter/parichheda
4. `धारा \d+[\.।\s]` — dhara (constitution-style)

Split strategy:
- Find all matches of any of the above patterns.
- Each match starts a new component.
- Text before the first match = preamble component (`component_type='full'`, `number='0'`).
- Strip `<amend>...</amend>` tags from `text_ne` before storing.
- If no matches at all: whole content = one component (`component_type='full'`, `number='1'`).
- Minimum component length: 20 characters after stripping. Skip shorter fragments.

Component URI: `{work_uri}/{component_type}/{number}`

---

## Ingestion script — `scripts/ingest_laws.py`

```
python3 scripts/ingest_laws.py [--limit N] [--offset N]
```

`--limit` and `--offset` for batch runs during development. Default: all 677.

**Pipeline per law:**

1. Parse law via `app/authority/parser.py`
2. Insert `work` (ON CONFLICT DO NOTHING on `uri`)
3. Insert `source_publication`:
   - `kind = 'official_copy_unverified'`
   - `sha256 = ParsedLaw.source_sha256`
   - `ocr_confidence = None` (no OCR — this is already text)
4. For each component:
   a. Insert `component` (ON CONFLICT DO NOTHING on `uri`)
   b. If `enactment_ad` is not None: insert `lifecycle_effect`:
      - `effect_type = 'commence'`
      - `legal_valid_time = f'[{enactment_ad},)'`
      - `transaction_time = '[now(),)'`
      - `effective_date = enactment_ad`
      - `commencement_dependency = NULL`
      - `approval_status = 'approved'`
      - `approved_by_1 = approved_by_2 = UUID('00000000-0000-0000-0000-000000000001')`
      - (**Phase 0 stub** — real dual approval is Phase A. Comment this clearly.)
   c. Insert `expression`: `as_of = enactment_ad`, `text_ne`, `text_hash`, `is_derived = True`
   d. Index to OpenSearch: `{component_uri, as_of, text_ne, dense_vector=[0.0*1536]}`

**Batching:** Commit to Postgres in batches of 100 components. Print progress every 50 laws.

**Error handling:** If a single law fails to parse or insert, log the error and continue. Do not abort the whole run.

---

## Retrieval modules

### `app/retrieval/eligibility_gate.py`
```python
def is_eligible(conn, component_uri: str, as_of: date) -> bool:
    """Calls is_eligible() SQL function."""
```

### `app/retrieval/dumb_retriever.py`
```python
def retrieve(query: str, as_of: date, k: int = 5) -> list[dict]:
    """
    OpenSearch BM25 match on text_ne, size=20.
    Filter each hit through is_eligible(component_uri, as_of).
    Return top k eligible: [{component_uri, text_ne, text_hash, score, work_title_ne}]
    No reranker. No embeddings.
    """
```

Include `work_title_ne` in the return by joining through the `component` and `work` tables
(look up `component_uri` → `work_id` → `work.title_ne`).

### `app/retrieval/validation_gate.py`
```python
def validate_and_render(
    claims: list[dict],   # [{"claim": str, "evidence_id": str}]
    as_of: date,
    conn,
) -> list[dict]:
    """
    For each evidence_id (= component_uri):
    1. Fetch expression WHERE component_uri=... AND as_of=... (exact match)
    2. Verify SHA-256(text_ne.encode()) == text_hash
    3. Verify is_eligible(component_uri, as_of)
    4. Pass → render citation from source_publication + work metadata
    5. Fail → abstain that claim (abstained=True, citation=None)
    """
```

Citation dict (from DB metadata, never from model output):
```python
{
    "component_uri": str,
    "work_title_ne": str,
    "work_title_en": str | None,
    "as_of": str,           # ISO date
    "source_kind": str,
    "ocr_confidence": float | None,
}
```

---

## `scripts/query.py`

```
python3 scripts/query.py "<question>" --as-of YYYY-MM-DD
```

Pipeline:
1. `dumb_retriever.retrieve(query, as_of, k=5)`
2. If no eligible hits → print "Abstaining — no eligible sources." Exit 0.
3. Build context string (top 5 hits, each prefixed with component_uri).
4. LangChain `ChatOpenAI(model='gpt-4o-mini', temperature=0.0)` with system prompt:
   ```
   You are Wakil-G. Answer using ONLY the provided context.
   Output JSON only:
   {"claims": [{"claim": "<text>", "evidence_id": "<component_uri>"}]}
   If context is insufficient: {"claims": [], "abstain": true}
   Do not write citations. Do not include anything not in the context.
   ```
5. Parse JSON output. Pass to `validate_and_render(claims, as_of, conn)`.
6. Print: answer claims + rendered citations.

---

## Eval gates — `app/eval/gates.py`

The test inserts temporary lifecycle_effect rows, runs retrieval, checks results, then deletes them. Use a fixed test component from any law that was successfully ingested (look up the first component_uri in the DB at runtime — don't hardcode sarbajanik).

```python
def check_repealed_as_current(conn, os_client) -> int:
    """
    1. Get first component_uri from component table.
    2. INSERT lifecycle_effect(effect_type='repeal', approval_status='approved',
       legal_valid_time='[2020-01-01,)') for that component.
    3. Verify is_eligible(component_uri, date(2024,1,1)) == False.
    4. DELETE the test row.
    Returns violation count (0 = gate works correctly).
    """

def check_not_yet_effective_as_current(conn, os_client) -> int:
    """
    1. Get second component_uri from component table.
    2. INSERT lifecycle_effect(effect_type='commence',
       commencement_dependency='gazette_notification',
       approval_status='approved', legal_valid_time='[2020-01-01,)').
    3. Verify is_eligible(component_uri, date(2024,1,1)) == False.
    4. DELETE the test row.
    Returns violation count.
    """
```

`__main__`:
```
repealed-as-current: <n>
not-yet-effective-as-current: <n>
```
Exit 1 if any n > 0.

Update `Makefile`:
```makefile
eval-gates:
	python3 -m app.eval.gates
```

---

## PS requirements mapped

| PS | Enforcement in this task |
|---|---|
| PS-2 | `commencement_dependency` blocks `is_eligible()` → `check_not_yet_effective_as_current` |
| PS-3 | Citations from `source_publication` + `work` metadata in `validation_gate.py` only |
| PS-6 | `as_of` explicit on every retrieve + validate call |
| PS-7 | Abstention owned by `validate_and_render()`; model `"abstain": true` is advisory input only |
| PS-15 | `EffectType.REPEAL`, `EXPIRY`, `DECLARED_INVALID` already in models.py |

---

## Zero-tolerance gates

- `repealed-as-current = 0` — `check_repealed_as_current()`
- `not-yet-effective-as-current = 0` — `check_not_yet_effective_as_current()`

Both must be 0 before merge. Run `make eval-gates` and include output.

---

## Required checks

```
make lint
make test
make eval-gates
```

---

## Commit authorship

Every commit MUST use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By: Claude`, no "Generated with Claude", no AI attribution.

---

## Return to Claude (via Prakash)

- Commit hash
- Changed/created files list
- `make lint` output
- `make test` output
- `make eval-gates` output
- `scripts/query.py "What are the powers of the Commission for Investigation of Abuse of Authority?" --as-of 2024-01-01` output
- Assumptions and remaining risks
