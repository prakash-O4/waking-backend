from __future__ import annotations

from typing import Any

from app.config import get_settings

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
    s = get_settings()
    if s.COHERE_API_KEY:
        try:
            import cohere

            co = cohere.Client(s.COHERE_API_KEY)
            docs = [h["text_ne"] for h in hits]
            response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=docs,
                top_n=k,
            )
            return [hits[r.index] for r in response.results]
        except Exception:
            pass

    try:
        return _flashrank_rerank(query, hits, k)
    except Exception:
        return hits[:k]
