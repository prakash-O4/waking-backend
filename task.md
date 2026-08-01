# Task P0-B — Phase 0 End-to-End: One Statute, One Query, One Citation

**Engineer:** Pi  
**Branch:** `phase-0/end-to-end`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Get one statute (sarbajanik) answerable end-to-end through the proper authority-store pipeline, with the eligibility gate and validation gate enforced, and `make eval-gates` confirming zero-tolerance gates are green. Do not touch `app/main.py` — build the new pipeline as parallel modules and wire it through `scripts/query.py`.

Phase 0 baseline: BM25-only retrieval, no reranker, no LangGraph. LangChain chain only.

---

## Acceptance criteria

1. `scripts/ingest_sarbajanik.py` runs without error (with `SUPABASE_DB_URL` + `OPENSEARCH_URL` set) and writes sarbajanik to Supabase + OpenSearch.
2. `scripts/query.py "question" --as-of 2024-10-08` returns an answer with citations rendered from canonical metadata (not from model output).
3. `make eval-gates` exits 0 and prints `repealed-as-current: 0 | not-yet-effective-as-current: 0`.
4. `make test` still green.
5. `make lint` still green.
6. The three P0-A known gaps are fixed (listed below).

---

## Fix first — P0-A known gaps

**F1 — `app/search/client.py`:** Replace `nori_tokenizer` (Korean) with try-order `icu_tokenizer` → `standard`:

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

**F3 — Eligibility gate:** Add `suspend` to NOT EXISTS clause. Do NOT re-run migration 001. Instead create `migrations/002_gate_suspend_fix.sql` containing only the updated `CREATE OR REPLACE FUNCTION is_eligible(...)` with `'suspend'` added to the effect_type IN list.

---

## Allowed scope

**Create:**
- `migrations/002_gate_suspend_fix.sql`
- `app/authority/writer.py`
- `app/retrieval/eligibility_gate.py`
- `app/retrieval/dumb_retriever.py`
- `app/retrieval/validation_gate.py`
- `app/eval/__init__.py`
- `app/eval/gates.py`
- `scripts/ingest_sarbajanik.py`
- `scripts/query.py`
- `tests/test_eligibility_gate.py`
- `tests/test_validation_gate.py`

**Modify:**
- `app/search/client.py` (F1)
- `app/authority/models.py` (F2)
- `scripts/migrate.py` (also apply migration 002)
- `Makefile` (wire `eval-gates` properly)

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
- Model must emit `claims + evidence_ids` only — no citations written by the model.
- No path skips the eligibility gate.
- No path bypasses the validation gate.

---

## Statute facts — sarbajanik

- **Title (ne):** सार्वजनिक प्रसारण सेवा ऐन, २०८१
- **Title (en):** Public Broadcasting Service Act 2081
- **URI:** `/np/act/2081/sarbajanik-prasaran-seva`
- **work_type:** `Act`
- **Enactment (AD):** `2024-10-08` — hardcoded. BS↔AD calendar is Phase C; do not call any library.
- **Status:** In force, no repeal, no commencement dependency.
- **Source data:** `app/processed/chunks/sarbajanik_chunks.json` (8 chapter chunks)
- **source_publication.kind:** `official_copy_unverified`

---

## Ingestion path — `scripts/ingest_sarbajanik.py`

All inserts idempotent (`INSERT ... ON CONFLICT DO NOTHING`). Steps in order:

**1. Register work** in `work` table.

**2. Register source_publication** — SHA-256 of `app/processed/markdown/sarbajanik.md` bytes, `kind='official_copy_unverified'`, `ocr_confidence=0.75`.

**3. Register components** — one per chunk, URI = `/np/act/2081/sarbajanik-prasaran-seva/parichheda/{chunk_id}`, `component_type='parichheda'`.

**4. Create lifecycle_effect (commence)** for each component:
- `effect_type='commence'`, `legal_valid_time='[2024-10-08,)'`, `transaction_time='[now(),)'`
- `effective_date=2024-10-08`, `commencement_dependency=NULL`
- `approval_status='approved'`
- `approved_by_1` = `approved_by_2` = `00000000-0000-0000-0000-000000000001` (sentinel — Phase 0 stub; comment this clearly)

**5. Write expressions** — one per chunk: `text_ne=chunk content`, `text_hash=SHA-256(text_ne.encode())`, `is_derived=True`, `as_of=2024-10-08`.

**6. Index to OpenSearch** — for each expression:
```json
{"component_uri": "...", "as_of": "2024-10-08", "text_ne": "...",
 "dense_vector": [0.0, ...(1536 zeros)]}
```
Zero vector placeholder — Phase 0 is BM25-only. Do not call any embeddings API.

---

## New retrieval modules

### `app/retrieval/eligibility_gate.py`
```python
def is_eligible(conn, component_uri: str, as_of: date) -> bool:
    # Calls is_eligible() SQL function. Returns bool.
```

### `app/retrieval/dumb_retriever.py`
```python
def retrieve(query: str, as_of: date, k: int = 5) -> list[dict]:
    # BM25 match query on text_ne, size=20.
    # Filter each hit through is_eligible().
    # Return top k eligible: [{component_uri, text_ne, text_hash, score}]
    # No reranker. No embeddings.
```

### `app/retrieval/validation_gate.py`
```python
def validate_and_render(
    claims: list[dict],   # [{"claim": str, "evidence_id": str}]
    as_of: date,
    conn,
) -> list[dict]:
    # For each evidence_id (= component_uri):
    #   1. Fetch expression WHERE component_uri=... AND as_of=...
    #   2. Verify SHA-256(text_ne) == text_hash
    #   3. Verify is_eligible(component_uri, as_of)
    #   4. Pass → render citation from source_publication metadata
    #   5. Fail → abstain that claim
    # Return [{claim, citation|None, abstained}]
```

Citation dict fields (from DB, not from model output):
```python
{"component_uri", "work_title_ne", "work_title_en", "as_of", "source_kind", "ocr_confidence"}
```

---

## `scripts/query.py`

CLI: `python3 scripts/query.py "<question>" --as-of YYYY-MM-DD`

Pipeline:
1. Retrieve via `dumb_retriever.retrieve(query, as_of)`
2. If no eligible hits: print "Abstaining — no eligible sources for this query and as-of." Exit.
3. Build context string from top hits.
4. LangChain prompt → LLM emits JSON: `{"claims": [{"claim": str, "evidence_id": str}]}` or `{"claims": [], "abstain": true}`
5. Pass claims to `validate_and_render()`
6. Print answer + citations.

LLM system prompt (do not change):
```
You are Wakil-G. Answer using ONLY the provided context.
Output JSON only: {"claims": [{"claim": "<text>", "evidence_id": "<component_uri>"}]}
Do not write citations. Do not include anything not in the context.
If context is empty or insufficient: {"claims": [], "abstain": true}
```

Model: `gpt-4o-mini`. Temperature: `0.0`.

---

## Eval gates — `app/eval/gates.py`

```python
def check_repealed_as_current(conn, os_client) -> int:
    # 1. INSERT lifecycle_effect(effect_type='repeal', approval_status='approved',
    #    legal_valid_time='[2024-10-08,)') for parichheda/0
    # 2. retrieve('broadcasting', as_of=date(2024,10,8))
    # 3. Count hits where is_eligible(component_uri, as_of) is False
    # 4. DELETE the test row
    # Returns violation count (must be 0)

def check_not_yet_effective_as_current(conn, os_client) -> int:
    # 1. INSERT lifecycle_effect(effect_type='commence',
    #    commencement_dependency='gazette_notification', approval_status='approved',
    #    legal_valid_time='[2024-10-08,)') for parichheda/1
    # 2. Verify is_eligible(parichheda/1 URI, date(2024,10,8)) == False
    # 3. DELETE the test row
    # Returns violation count (must be 0)
```

`__main__` in `app/eval/gates.py`:
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

| PS | How enforced in this task |
|---|---|
| PS-2 | `commencement_dependency` blocks `is_eligible()` → tested by `check_not_yet_effective_as_current` |
| PS-3 | Citations rendered from `source_publication` metadata in `validation_gate.py`, never from model output |
| PS-6 | `as_of` passed explicitly to every retrieval + validation call |
| PS-7 | Abstention owned by `validation_gate.py`; model self-abstention is advisory input only |
| PS-15 | `EffectType` enum has repeal/expiry/declared_invalid (already in models.py) |

---

## Zero-tolerance gates guarded

- `repealed-as-current = 0` — `check_repealed_as_current()`
- `not-yet-effective-as-current = 0` — `check_not_yet_effective_as_current()`

Both must be 0 before merge. Run `make eval-gates` and include output in return.

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
No `Co-Authored-By: Claude`, no "Generated with Claude", no AI attribution of any kind.

---

## Return to Claude (via Prakash)

- Commit hash
- Changed/created files list
- `make lint` output
- `make test` output
- `make eval-gates` output
- `scripts/query.py "What are the functions of the public broadcasting institution?" --as-of 2024-10-08` output
- Assumptions and remaining risks
