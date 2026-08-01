from __future__ import annotations

from datetime import date
from typing import Any

from psycopg2.extensions import connection

from app.authority.writer import connect
from app.retrieval.eligibility_gate import is_eligible
from app.search.client import DEFAULT_INDEX, get_client


def _work_titles(conn: connection, component_uri: str) -> tuple[str | None, str | None]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT w.title_ne, w.title_en
            FROM component c JOIN work w ON w.id = c.work_id
            WHERE c.uri = %s
            """,
            (component_uri,),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)


def retrieve(query: str, as_of: date, k: int = 5) -> list[dict[str, Any]]:
    """BM25 over OpenSearch, then Postgres eligibility filter. No reranker."""
    client = get_client()
    response = client.search(
        index=DEFAULT_INDEX,
        body={"query": {"match": {"text_ne": query}}, "size": 20},
    )
    results: list[dict[str, Any]] = []
    with connect() as conn:
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            component_uri = source.get("component_uri")
            if not component_uri or not is_eligible(conn, component_uri, as_of):
                continue
            title_ne, _title_en = _work_titles(conn, component_uri)
            results.append(
                {
                    "component_uri": component_uri,
                    "text_ne": source.get("text_ne", ""),
                    "text_hash": source.get("text_hash", ""),
                    "score": hit.get("_score", 0.0),
                    "work_title_ne": title_ne,
                }
            )
            if len(results) >= k:
                break
    return results
