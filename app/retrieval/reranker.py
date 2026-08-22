from __future__ import annotations

from typing import Any


def rerank(query: str, hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Cohere reranker; no-op unless configured and installed."""
    from app.config import get_settings

    s = get_settings()
    if not s.COHERE_API_KEY:
        return hits[:k]
    try:
        import cohere
    except ImportError:
        return hits[:k]

    co = cohere.Client(s.COHERE_API_KEY)
    docs = [h["text_ne"] for h in hits]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs,
        top_n=k,
    )
    return [hits[r.index] for r in response.results]
