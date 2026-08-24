# Task: RET-C — FlashRank fallback reranker

**Branch:** `ret/flashrank-fallback`  
**Base:** `dev`  
**Engineer:** Pi  

---

## Objective

The current `reranker.py` has no exception handling around the Cohere API call. Any rate-limit error, 503, or transient failure propagates uncaught to `retrieve_postgres`. The system design's degraded-mode ladder (§9) requires the reranker path to degrade gracefully.

**Fix:** wrap the Cohere call in `try/except`. On any exception, fall through to **FlashRank** (`ms-marco-MultiBERT-L-12` — multilingual, supports Devanagari), which runs locally with no API key. If FlashRank is also unavailable (import error or model failure), fall through to the existing passthrough (`hits[:k]`).

Priority ladder: **Cohere → FlashRank → passthrough.**

---

## Acceptance criteria

1. Cohere is tried first when `COHERE_API_KEY` is set.
2. Any Cohere exception (rate limit, API error, network error) falls through to FlashRank — no exception propagates to the caller.
3. When `COHERE_API_KEY` is unset, FlashRank is used directly (no Cohere attempt).
4. Any FlashRank failure (import error, model error) falls through to `hits[:k]` passthrough.
5. The `Ranker` object is cached at module level — not recreated on every call.
6. `make test` green, `make lint` clean.

---

## Exact scope

**Only these files may change:**

- `requirements.txt` — add `flashrank`
- `app/retrieval/reranker.py` — add fallback logic and Ranker cache
- `tests/test_retrieval.py` — update existing reranker test, add new tests

Do NOT modify `postgres_retriever.py`, `config.py`, `gated_orchestrator.py`, or any other file. No new files.

---

## Implementation guide

### 1. `requirements.txt`

Add one line (near other retrieval-related packages):

```
flashrank
```

No version pin — FlashRank does not conflict with existing deps.

### 2. `app/retrieval/reranker.py` — full rewrite

```python
from __future__ import annotations

from typing import Any

_ranker: Any = None


def _get_ranker() -> Any:
    global _ranker
    if _ranker is None:
        from flashrank import Ranker  # type: ignore[import-not-found]
        _ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
    return _ranker


def _flashrank_rerank(
    query: str, hits: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    from flashrank import RerankRequest  # type: ignore[import-not-found]

    ranker = _get_ranker()
    passages = [{"id": i, "text": h["text_ne"]} for i, h in enumerate(hits)]
    req = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(req)
    return [hits[r["id"]] for r in results[:k]]


def rerank(query: str, hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    from app.config import get_settings

    if get_settings().COHERE_API_KEY:
        try:
            import cohere

            co = cohere.Client(get_settings().COHERE_API_KEY)
            docs = [h["text_ne"] for h in hits]
            response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=docs,
                top_n=k,
            )
            return [hits[r.index] for r in response.results]
        except Exception:
            pass  # fall through to FlashRank

    try:
        return _flashrank_rerank(query, hits, k)
    except Exception:
        return hits[:k]
```

Key points:
- `_ranker` is module-level — `_get_ranker()` loads `ms-marco-MultiBERT-L-12` once and caches it.
- Cohere is only attempted when `COHERE_API_KEY` is set.
- `except Exception: pass` on Cohere silently falls through.
- `except Exception: return hits[:k]` on FlashRank is the final passthrough fallback.

---

## Tests required (`tests/test_retrieval.py`)

**Delete** the existing `test_reranker_skipped_when_cohere_key_unset` test — it is semantically wrong after this change (FlashRank now runs when key is unset). Replace it with these four tests:

```python
def test_cohere_reranks_when_key_set(monkeypatch: Any) -> None:
    import types, sys

    class Settings:
        COHERE_API_KEY = "key"

    class FakeResult:
        index = 1

    class FakeResponse:
        results = [FakeResult()]

    class FakeClient:
        def __init__(self, key: str) -> None:
            pass
        def rerank(self, **kwargs: Any) -> FakeResponse:
            return FakeResponse()

    cohere_mod = types.ModuleType("cohere")
    cohere_mod.Client = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", cohere_mod)
    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b"}]  # index=1


def test_flashrank_fallback_when_cohere_raises(monkeypatch: Any) -> None:
    import types, sys

    class Settings:
        COHERE_API_KEY = "key"

    class BadClient:
        def __init__(self, key: str) -> None:
            pass
        def rerank(self, **kwargs: Any) -> None:
            raise RuntimeError("rate limited")

    cohere_mod = types.ModuleType("cohere")
    cohere_mod.Client = BadClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", cohere_mod)
    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr(
        "app.retrieval.reranker._flashrank_rerank", lambda q, h, k: [h[1]]
    )

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b"}]


def test_flashrank_used_when_no_cohere_key(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = ""

    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr(
        "app.retrieval.reranker._flashrank_rerank", lambda q, h, k: [h[1]]
    )

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "b"}]


def test_passthrough_when_both_fail(monkeypatch: Any) -> None:
    class Settings:
        COHERE_API_KEY = ""

    def _bad_flashrank(q: str, h: list, k: int) -> list:
        raise RuntimeError("model error")

    monkeypatch.setattr("app.retrieval.reranker.get_settings", lambda: Settings())
    monkeypatch.setattr("app.retrieval.reranker._flashrank_rerank", _bad_flashrank)

    hits = [{"text_ne": "a"}, {"text_ne": "b"}]
    assert rerank("q", hits, 1) == [{"text_ne": "a"}]
```

Note: `rerank` is already imported at the top of the test file — no new import needed.

---

## System Design refs

- §9 Degraded-mode ladder: "Reranker down → BM25-only path, flagged." FlashRank is a local reranker that sits between Cohere and BM25-only — it improves on the bare degraded mode without requiring network access.
- §2 Core Invariant #2 — eligibility gate untouched; this change is entirely post-retrieval.
- No PS requirements are in scope for this change.

---

## Zero-tolerance gates guarded

None in scope. This change cannot affect `repealed-as-current`, `not-yet-effective-as-current`, or `overruled-as-good-law` (reranking is post-retrieval, post-eligibility-gate).

---

## Required checks

```bash
make test
make lint
```

---

## Commit authorship

Every commit must be authored as:

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No AI attribution. No `Co-Authored-By` trailers. No "Generated with Claude" lines.

---

## Return to Claude

When done, return: commit hash, list of changed files, output of `make test` and `make lint`, and any assumptions made.
