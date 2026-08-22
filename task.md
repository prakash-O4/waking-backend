# Task CLEANUP-A: Remove OpenSearch

**Engineer:** Pi  
**Branch:** `cleanup/remove-opensearch`  
**Base:** `dev` (commit bd3870b)

---

## Objective

OpenSearch is unused dead infrastructure — all ingested data lives in
pgvector, the query plane already falls back to `retrieve_postgres` on every
call, and the Docker container consumes 512 MB of heap for nothing.

Remove it entirely. Postgres is the primary retriever going forward.

---

## Exact changes required

### 1. Delete these files completely

- `app/search/client.py`
- `app/search/__init__.py`
- `app/retrieval/dumb_retriever.py`

### 2. `app/retrieval/gated_orchestrator.py`

- Remove: `from app.retrieval.dumb_retriever import retrieve as os_retrieve`
- Delete the entire `_try_retrieve()` function (lines that import opensearchpy,
  catch ConnectionError/TransportError, and fall back to retrieve_postgres)
- In `answer()`, replace:
  ```python
  hits = _try_retrieve(conn, subquery_text, subquery_as_of)
  ```
  with:
  ```python
  hits = retrieve_postgres(conn, subquery_text, subquery_as_of)
  ```

### 3. `requirements.txt`

Remove this line:
```
opensearch-py==2.7.1
```

### 4. `docker-compose.yml`

Remove the entire `opensearch:` service block. If the file is empty after,
delete it.

### 5. `Makefile` — `setup` target

Remove the opensearch Docker startup block:
```
docker compose up -d opensearch
for i in ... curl ... sleep ... done
curl -fsS ... >/dev/null
python3 -m app.search.client
```
Keep the rest of `make setup` intact.

### 6. `tests/test_degraded_modes.py`

- Remove `import opensearchpy` (line 7)
- Delete `test_opensearch_down_uses_postgres_fallback` entirely
- Delete `test_opensearch_down_and_postgres_empty_abstains` entirely
- In `test_model_down_uses_extractive_claim_still_validated`, change:
  ```python
  monkeypatch.setattr(orchestrator, "os_retrieve", lambda query, as_of, k: [hit])
  ```
  to:
  ```python
  monkeypatch.setattr(orchestrator, "retrieve_postgres", lambda conn, q, a, k: [hit])
  ```
- Keep `test_postgres_down_returns_503` and
  `test_model_down_uses_extractive_claim_still_validated` (fixed as above)

### 7. `app/eval/romanized_slice.py`

Replace:
```python
from app.retrieval.dumb_retriever import retrieve
# ...
results = retrieve(entry["query"], as_of, k=k)
```
With:
```python
from app.authority.writer import connect
from app.retrieval.postgres_retriever import retrieve_postgres
# ...
with connect() as conn:
    results = retrieve_postgres(conn, entry["query"], as_of, k=k)
```

### 8. `app/eval/phase_c_slice.py`

Same pattern — replace `dumb_retriever.retrieve(query, as_of)` with:
```python
from app.authority.writer import connect
from app.retrieval.postgres_retriever import retrieve_postgres
# ...
with connect() as conn:
    hits = retrieve_postgres(conn, query, as_of)
```

---

## What does NOT change

- `app/retrieval/postgres_retriever.py` — untouched
- `app/retrieval/eligibility_gate.py` — untouched
- `app/retrieval/validation_gate.py` — untouched
- All ingestion pipeline files — untouched
- All eval metrics and golden sets — untouched
- `tests/test_postgres_down_returns_503` — kept, still valid

---

## Required checks

```
make test
make lint
```

`make test` must show **35 passed** (same count as before — the 2 deleted
OpenSearch tests were already failing and are not in the passing count).

Return commit hash, checks run/results.

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By`, no AI attribution.
