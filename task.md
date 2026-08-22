# Task OBS-RET: Retrieval observability + eval slice

**Engineer:** Pi  
**Branch:** `feat/obs-retrieval`  
**Base:** `dev`

---

## Objective

Three gaps to close in one task:

1. **Retrieval is a black box** — `retrieve_postgres` runs 6 internal stages with
   zero visibility into latency, counts, or scores.

2. **Answer trace is flat** — `gated_orchestrator` emits one trace at the end with
   total latency only. No per-stage breakdown, no validation gate outcome, no
   per-chunk scores.

3. **Eval slice is broken** — `romanized_slice.py` matches on `/np/act/...` URI
   prefixes. The new retriever returns chunk UUIDs. Every eval run returns 0 hits.

---

## Target trace shape (Langfuse)

```
rag.answer  (trace)
  metadata:
    query_hash · as_of · query_type · result_count · abstained
    retrieval_latency_ms · generation_latency_ms · validation_latency_ms
    top_chunk_scores · validation_claims_passed · validation_claims_abstained

  rag.retrieval (trace — emitted by retrieve_postgres)
    stage.eligibility_gate:  eligible_count · latency_ms
    stage.vector_search:     candidate_count · top_score · latency_ms
    stage.lexical_search:    ran · candidate_count · latency_ms
    stage.rrf_fusion:        merged_count · top_rrf_score · latency_ms
    stage.relevance_gate:    passed_count · abstained · latency_ms
    stage.rerank:            ran · final_count · top_score · latency_ms
```

PS-14: traces store `query_hash` (SHA-256), never raw query text or chunk text.

---

## Files in scope

| File | Action |
|---|---|
| `app/retrieval/postgres_retriever.py` | Add Langfuse stage spans |
| `app/retrieval/gated_orchestrator.py` | Expand answer trace with stage timings + outcomes |
| `app/eval/retrieval_slice.py` | New — Recall@1/3/5 + MRR |
| `app/eval/romanized_slice.py` | Fix broken URI matching |
| `Makefile` | Wire `retrieval_slice` into `make eval` |

**Do NOT touch:** eligibility_gate, validation_gate, reranker, ingestion pipeline,
chunkers, RAGAS slices, golden JSON files, tests (no new tests required — existing
42 still pass because tracing is a no-op when LANGFUSE_PUBLIC_KEY is unset).

---

## 1. `app/retrieval/postgres_retriever.py` — add tracing

### Module-level Langfuse singleton (same pattern as `pipeline.py`)

```python
import hashlib
import time
from typing import Any

_lf_client: Any = None

def _get_lf_client() -> Any | None:
    from app.config import get_settings
    if not get_settings().LANGFUSE_PUBLIC_KEY:
        return None
    global _lf_client
    if _lf_client is None:
        try:
            from langfuse import Langfuse
        except ImportError:
            return None
        s = get_settings()
        _lf_client = Langfuse(
            public_key=s.LANGFUSE_PUBLIC_KEY,
            secret_key=s.LANGFUSE_SECRET_KEY,
            host=s.LANGFUSE_HOST,
        )
    return _lf_client

def _span(trace: Any, stage: str, **metadata: Any) -> None:
    if trace is None:
        return
    trace.span(name=f"stage.{stage}", metadata=metadata)
```

### In `retrieve_postgres` — wrap each stage with timing and spans

At function start:
```python
lf = _get_lf_client()
qhash = hashlib.sha256(query.encode()).hexdigest()
trace = lf.trace(name="rag.retrieval", metadata={"query_hash": qhash, "as_of": str(as_of), "k": k}) if lf else None
```

After each stage (pattern):
```python
t0 = time.monotonic()
# ... stage work ...
_span(trace, "STAGE_NAME", latency_ms=int((time.monotonic()-t0)*1000), **counts_and_scores)
```

**Stages and their metadata fields:**

| Stage name | metadata fields |
|---|---|
| `eligibility_gate` | `eligible_count`, `latency_ms` |
| `vector_search` | `candidate_count`, `top_score` (first row's vec_score or 0.0), `latency_ms` |
| `lexical_search` | `ran` (bool), `candidate_count`, `latency_ms` |
| `rrf_fusion` | `merged_count`, `top_rrf_score` (first score after sort), `latency_ms` |
| `relevance_gate` | `passed_count`, `abstained` (bool), `latency_ms` |
| `rerank` | `ran` (bool, true if COHERE_API_KEY set), `final_count`, `latency_ms` |

At every return path:
```python
if lf:
    lf.flush()
```

Early returns (empty eligible, no candidates after relevance gate) must still
flush before returning.

---

## 2. `app/retrieval/gated_orchestrator.py` — expand answer trace

### Stage timing

Wrap each major call with `time.monotonic()`:

```python
# retrieval
t_ret = time.monotonic()
hits = retrieve_postgres(conn, subquery_text, subquery_as_of)
retrieval_latency_ms += int((time.monotonic() - t_ret) * 1000)

# generation (inside _model_claims call)
t_gen = time.monotonic()
claims_result = _model_claims(question, all_hits)
generation_latency_ms = int((time.monotonic() - t_gen) * 1000)

# validation
t_val = time.monotonic()
rendered = validate_and_render(claims, session_as_of, conn)
validation_latency_ms = int((time.monotonic() - t_val) * 1000)
```

`retrieval_latency_ms` is cumulative across subqueries (sum across the loop).

### Validation gate outcome — compute from rendered results (no signature change)

```python
claims_passed = sum(1 for r in rendered if not r.get("abstained"))
claims_abstained = sum(1 for r in rendered if r.get("abstained"))
```

### Top chunk scores

```python
top_chunk_scores = sorted(
    [h.get("score", 0.0) for h in all_hits], reverse=True
)[:5]
```

### Expand `_emit_answer_trace` metadata dict

Add these keys to the metadata passed to `_emit_answer_trace`:
```python
"retrieval_latency_ms": retrieval_latency_ms,
"generation_latency_ms": generation_latency_ms,
"validation_latency_ms": validation_latency_ms,
"validation_claims_passed": claims_passed,
"validation_claims_abstained": claims_abstained,
"top_chunk_scores": top_chunk_scores,
```

Keep all existing keys unchanged (`query_hash`, `as_of`, `query_type`,
`latency_ms`, `retrieved_uris`, `gate_decision`, `result_count`).

---

## 3. Fix `retrieve_postgres` — add `document_source_id` to hit dict

The eval slices need to match retrieved chunks back to source documents.
In the **final result assembly query** (the `SELECT ... WHERE id::text = ANY(%(ids)s)`),
add a JOIN to `documents`:

```sql
SELECT c.id::text, c.chunk_text, c.span_sha256, c.act_name, c.case_id,
       c.chunk_type, c.section_number, d.source_id
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.id::text = ANY(%(ids)s)
```

Update `_hit()` to accept and include `document_source_id`:

```python
def _hit(row: tuple[Any, ...], score: float) -> dict[str, Any]:
    chunk_id, text, text_hash, act_name, case_id, chunk_type, section_number, source_id = row[:8]
    return {
        "component_uri": str(chunk_id),
        "text_ne": text,
        "text_hash": text_hash,
        "score": float(score),
        "work_title_ne": act_name or case_id or "",
        "chunk_type": chunk_type,
        "section_number": section_number or "",
        "document_source_id": str(source_id) if source_id else "",
    }
```

---

## 4. `app/eval/romanized_slice.py` — fix URI matching

Current broken code matches `component_uri.startswith(expected_prefix)`.
New retriever returns chunk UUIDs — prefix match never fires.

The golden set `expected_uris` have format `/np/act/{year}/{source_id}`.
The source_id is the last path segment.

**New matching logic:**

```python
expected_source_ids = {
    uri.split("/")[-1]
    for entry in golden
    for uri in entry.get("expected_uris", [])
}

# In the loop, after retrieval:
result_source_ids = [r.get("document_source_id", "") for r in results]
if any(sid in expected_source_ids_for_entry for sid in result_source_ids):
    hits += 1
```

Where `expected_source_ids_for_entry` is the set of source_ids for the
current golden entry (not global).

Full corrected loop:

```python
hits = 0
for entry in golden:
    as_of = date.fromisoformat(entry["as_of"])
    expected_sids = {u.split("/")[-1] for u in entry.get("expected_uris", [])}
    with connect() as conn:
        results = retrieve_postgres(conn, entry["query"], as_of, k=k)
    result_sids = [r.get("document_source_id", "") for r in results]
    if any(sid in expected_sids for sid in result_sids):
        hits += 1
```

Remove the old `startswith` prefix logic entirely.

---

## 5. `app/eval/retrieval_slice.py` — new file

Recall@K and MRR against the same golden set.

```python
"""
Retrieval quality eval: Recall@1/3/5 and MRR.
Uses golden/romanized.json — same queries, matching on document_source_id.

Run: python3 -m app.eval.retrieval_slice
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.authority.writer import connect
from app.retrieval.postgres_retriever import retrieve_postgres

_GOLDEN = Path(__file__).parent / "golden" / "romanized.json"


def run_slice(ks: list[int] | None = None) -> dict[str, Any]:
    if ks is None:
        ks = [1, 3, 5]
    golden: list[dict[str, Any]] = json.loads(_GOLDEN.read_text())
    max_k = max(ks)

    recall: dict[int, int] = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []

    for entry in golden:
        as_of = date.fromisoformat(entry["as_of"])
        expected_sids = {u.split("/")[-1] for u in entry.get("expected_uris", [])}
        with connect() as conn:
            results = retrieve_postgres(conn, entry["query"], as_of, k=max_k)

        result_sids = [r.get("document_source_id", "") for r in results]

        # Recall@K
        for k in ks:
            if any(sid in expected_sids for sid in result_sids[:k]):
                recall[k] += 1

        # MRR — rank of first relevant result
        rr = 0.0
        for rank, sid in enumerate(result_sids, start=1):
            if sid in expected_sids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(golden)
    return {
        **{f"recall_at_{k}": recall[k] / n for k in ks},
        "mrr": sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0,
        "n": n,
    }


if __name__ == "__main__":
    result = run_slice()
    print(json.dumps(result, indent=2))
```

---

## 6. `Makefile` — wire into `make eval`

In the `eval` target, add:
```makefile
python3 -m app.eval.retrieval_slice
```

alongside the existing `python3 -m app.eval.romanized_slice`.

---

## Required checks

```
make test    # must still show 42 passed, 2 skipped
make lint
```

Tracing is a no-op when `LANGFUSE_PUBLIC_KEY` is unset — all 42 existing tests
pass unchanged. No new tests required for this task.

**Live smoke test (run locally, not part of make test):**
```
LANGFUSE_PUBLIC_KEY= python3 -m app.eval.retrieval_slice
```
Must return non-zero `recall_at_5` and `mrr` against the 200 ingested laws.

Return commit hash, changed files, checks run/results, and the retrieval_slice
output from the smoke test.

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No `Co-Authored-By`, no AI attribution.
