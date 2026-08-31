from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.ingestion.metadata_enricher import _apply_chunk_metadata, _parse_json


def chunks(*indexes: int) -> list[Any]:
    return [SimpleNamespace(chunk_index=i) for i in indexes]


def test_parse_json_strips_markdown_fence() -> None:
    assert _parse_json('```json\n[{"chunk_index": 0}]\n```', list) == [
        {"chunk_index": 0}
    ]


@pytest.mark.parametrize("raw", ["[", "not json", '{"summary": "x"}'])
def test_apply_chunk_metadata_parse_failures_fall_back(raw: str) -> None:
    assert _apply_chunk_metadata(chunks(0), raw) == [
        {"chunk_index": 0, "keywords": None, "relevant_questions": None}
    ]


def test_apply_chunk_metadata_skips_missing_or_wrong_typed_chunk_index() -> None:
    raw = """
    [
      {"keywords": ["missing"]},
      {"chunk_index": "0", "keywords": ["wrong"]},
      {"chunk_index": 0, "keywords": ["right"], "relevant_questions": ["q"]}
    ]
    """
    assert _apply_chunk_metadata(chunks(0), raw) == [
        {"chunk_index": 0, "keywords": ["right"], "relevant_questions": ["q"]}
    ]


@pytest.mark.parametrize("value", ["not-a-list", {"nested": "dict"}, None])
def test_apply_chunk_metadata_wrong_shaped_values_keep_current_normalization(
    value: Any,
) -> None:
    out = _apply_chunk_metadata(
        chunks(0),
        json.dumps(
            [{"chunk_index": 0, "keywords": value, "relevant_questions": value}]
        ),
    )
    expected = value or None
    assert out == [
        {"chunk_index": 0, "keywords": expected, "relevant_questions": expected}
    ]


def test_apply_chunk_metadata_partial_failure_isolated() -> None:
    raw = """
    [
      {"chunk_index": "bad", "keywords": ["bad"]},
      {"chunk_index": 1, "keywords": ["good"], "relevant_questions": ["q"]}
    ]
    """
    assert _apply_chunk_metadata(chunks(0, 1), raw) == [
        {"chunk_index": 0, "keywords": None, "relevant_questions": None},
        {"chunk_index": 1, "keywords": ["good"], "relevant_questions": ["q"]},
    ]
