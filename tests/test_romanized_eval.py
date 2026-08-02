from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.eval import romanized_slice


def _golden(tmp_path: Path, expected_uri: str = "/np/act/2059/foo") -> Path:
    path = tmp_path / "romanized.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "bhrastachar niwaran ain",
                    "as_of": "2024-01-01",
                    "expected_uris": [expected_uri],
                }
            ]
        )
    )
    return path


def test_run_slice_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(romanized_slice, "_GOLDEN", _golden(tmp_path))
    monkeypatch.setattr(
        romanized_slice,
        "retrieve",
        lambda *_args, **_kwargs: [{"component_uri": "/np/act/2059/foo"}],
    )
    assert romanized_slice.run_slice()["recall_at_k"] == 1.0


def test_run_slice_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(romanized_slice, "_GOLDEN", _golden(tmp_path))
    monkeypatch.setattr(romanized_slice, "retrieve", lambda *_args, **_kwargs: [])
    assert romanized_slice.run_slice()["recall_at_k"] == 0.0


def test_run_slice_prefix_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(romanized_slice, "_GOLDEN", _golden(tmp_path))
    monkeypatch.setattr(
        romanized_slice,
        "retrieve",
        lambda *_args, **_kwargs: [{"component_uri": "/np/act/2059/foo/dafa/1"}],
    )
    result = romanized_slice.run_slice()
    assert result["hits"] == 1


def test_run_slice_runtime_error_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(romanized_slice, "_GOLDEN", _golden(tmp_path))
    monkeypatch.setattr(romanized_slice, "retrieve", fail)
    with pytest.raises(RuntimeError, match="db unavailable"):
        romanized_slice.run_slice()
