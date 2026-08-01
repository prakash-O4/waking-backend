from __future__ import annotations

import os
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import RequestError


DEFAULT_INDEX = "wakilg_authority"


def get_client() -> OpenSearch:
    return OpenSearch(os.getenv("OPENSEARCH_URL", "http://localhost:9200"))


def _mapping(tokenizer: str) -> dict[str, Any]:
    return {
        "settings": {
            "index": {"knn": True},
            "analysis": {
                "analyzer": {
                    "ne_text": {
                        "type": "custom",
                        "tokenizer": tokenizer,
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "component_uri": {"type": "keyword"},
                "as_of": {"type": "date"},
                "text_ne": {"type": "text", "analyzer": "ne_text"},
                "dense_vector": {"type": "knn_vector", "dimension": 1536},
            }
        },
    }


def ensure_index(index_name: str = DEFAULT_INDEX) -> None:
    client = get_client()
    if client.indices.exists(index=index_name):
        return
    try:
        client.indices.create(index=index_name, body=_mapping("nori_tokenizer"))
    except RequestError:
        if client.indices.exists(index=index_name):
            return
        client.indices.create(index=index_name, body=_mapping("standard"))


if __name__ == "__main__":
    ensure_index()
    print("OpenSearch reachable; index ready")
