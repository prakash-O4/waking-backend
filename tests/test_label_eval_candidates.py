from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.label_eval_candidates as lec


class Client:
    def __init__(self, traces: list[Any]) -> None:
        self.traces = traces
        self.api = SimpleNamespace(trace=SimpleNamespace(list=self._list))

    def _list(self, **kwargs: Any) -> Any:
        return SimpleNamespace(data=self.traces)


def trace(
    id: str, input: Any = "real question", as_of: str | None = "2024-01-02"
) -> Any:
    return SimpleNamespace(
        id=id,
        input=input,
        metadata={} if as_of is None else {"as_of": as_of},
        timestamp=datetime(2024, 2, 3, tzinfo=timezone.utc),
    )


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lec, "QUEUE_FILE", tmp_path / "_traffic_queue.json")
    monkeypatch.setattr(lec, "LABELED_TRAFFIC_FILE", tmp_path / "labeled_traffic.json")
    monkeypatch.setattr(lec, "CLAIM_SUPPORT_FILE", tmp_path / "claim_support.json")
    return tmp_path


def read(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def seed_queue(files: Path, **extra: Any) -> None:
    row = {
        "trace_id": "t1",
        "question": "question?",
        "as_of": "2024-01-02",
        "as_of_source": "metadata",
        "content_available": True,
        "fetched_at": "now",
        "status": "pending",
    }
    row.update(extra)
    (files / "_traffic_queue.json").write_text(json.dumps([row]), encoding="utf-8")


class Conn:
    def __enter__(self) -> "Conn":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    claims: list[dict[str, Any]] | None = None,
    expressions: dict[str, tuple[str, str] | None] | None = None,
) -> None:
    monkeypatch.setattr(lec, "connect", Conn)
    monkeypatch.setattr(
        lec,
        "retrieve_postgres",
        lambda conn, q, as_of: [{"component_uri": "u1", "text_ne": "txt", "tier": 1}],
    )
    claims = claims or [{"claim": "c0", "quote": "q0", "evidence_id": "u1"}]
    monkeypatch.setattr(
        lec, "_structured_claims", lambda *a: {"claims": claims, "abstain": False}
    )
    monkeypatch.setattr(
        lec,
        "validate_and_render",
        lambda claims, as_of, conn: [
            {"claim": c["claim"], "abstained": i == 0} for i, c in enumerate(claims)
        ],
    )
    expressions = expressions or {}
    monkeypatch.setattr(
        lec,
        "_expression",
        lambda conn, evidence_id, as_of: expressions.get(
            evidence_id, ("default expression text", "sha")
        ),
    )


def test_fetch_dedupes_trace_id(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lec, "get_lf_client", lambda: Client([trace("t1")]))
    lec.fetch(7, 50)
    lec.fetch(7, 50)
    assert [r["trace_id"] for r in read(files / "_traffic_queue.json")] == ["t1"]


def test_fetch_marks_hash_content_unavailable(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lec,
        "get_lf_client",
        lambda: Client([trace("hash", "abcdef0123456789"), trace("real", "hello?")]),
    )
    lec.fetch(7, 50)
    rows = {r["trace_id"]: r for r in read(files / "_traffic_queue.json")}
    assert rows["hash"]["content_available"] is False
    assert rows["real"]["content_available"] is True


def test_fetch_as_of_metadata_and_fallback(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lec,
        "get_lf_client",
        lambda: Client([trace("m", as_of="2024-01-02"), trace("f", as_of=None)]),
    )
    lec.fetch(7, 50)
    rows = {r["trace_id"]: r for r in read(files / "_traffic_queue.json")}
    assert rows["m"]["as_of"] == "2024-01-02"
    assert rows["m"]["as_of_source"] == "metadata"
    assert rows["f"]["as_of"] == "2024-02-03"
    assert rows["f"]["as_of_source"] == "trace_timestamp_fallback"


def test_show_and_label_refuse_unavailable_without_pipeline(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files, content_available=False, question="abcdef0123456789")
    monkeypatch.setattr(
        lec, "retrieve_postgres", lambda *a: pytest.fail("pipeline called")
    )
    with pytest.raises(SystemExit):
        lec.show("t1")
    with pytest.raises(SystemExit):
        lec.label("t1", "Prakash", None, ["0:supports"])


def test_label_rejects_out_of_range_claim(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    patch_pipeline(monkeypatch, [{"claim": "c", "quote": "q", "evidence_id": "u"}])
    with pytest.raises(SystemExit):
        lec.label("t1", "Prakash", None, ["1:supports"])


def test_label_writes_uris_only_claims_only_and_requires_one(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    patch_pipeline(monkeypatch)
    lec.label("t1", "Prakash", "u1,u2", None)
    assert read(files / "labeled_traffic.json")[0]["expected_uris"] == ["u1", "u2"]
    assert not (files / "claim_support.json").exists()

    seed_queue(files, status="pending")
    lec.label("t1", "Prakash", None, ["0:unsupported"])
    assert read(files / "claim_support.json")[0]["supports"] is False

    with pytest.raises(SystemExit):
        lec.label("t1", "Prakash", None, None)


def test_label_quote_check_passed_true(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    quote = "यो लामो उद्धरण पाठ हो"
    claims = [{"claim": "c", "quote": quote, "evidence_id": "u"}]
    patch_pipeline(monkeypatch, claims, {"u": (f"अगाडि {quote} पछाडि", "sha")})
    lec.label("t1", "Prakash", None, ["0:supports"])
    assert read(files / "claim_support.json")[0]["quote_check_passed"] is True


def test_label_quote_check_passed_false_when_quote_missing(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    claims = [{"claim": "c", "quote": "यो लामो उद्धरण पाठ हो", "evidence_id": "u"}]
    patch_pipeline(monkeypatch, claims, {"u": ("अर्कै पाठ", "sha")})
    lec.label("t1", "Prakash", None, ["0:unsupported"])
    assert read(files / "claim_support.json")[0]["quote_check_passed"] is False


def test_label_quote_check_passed_false_when_expression_missing(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    claims = [{"claim": "c", "quote": "यो लामो उद्धरण पाठ हो", "evidence_id": "u"}]
    patch_pipeline(monkeypatch, claims, {"u": None})
    lec.label("t1", "Prakash", None, ["0:unsupported"])
    assert read(files / "claim_support.json")[0]["quote_check_passed"] is False


def test_report_empty(files: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lec.report()
    assert "no labeled claims yet" in capsys.readouterr().out


def test_report_counts_gap_rate_and_examples(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {"supports": True, "quote_check_passed": True, "source": "langfuse:tt"},
        {"supports": True, "quote_check_passed": False, "source": "langfuse:tf"},
        {"supports": False, "quote_check_passed": True, "source": "langfuse:ft1"},
        {"supports": False, "quote_check_passed": True, "source": "langfuse:ft2"},
        {"supports": False, "quote_check_passed": False, "source": "langfuse:ff"},
    ]
    (files / "claim_support.json").write_text(json.dumps(rows), encoding="utf-8")
    lec.report()
    out = capsys.readouterr().out
    assert "supports=True, quote_check_passed=True: 1" in out
    assert "supports=True, quote_check_passed=False: 1" in out
    assert "supports=False, quote_check_passed=True: 2" in out
    assert "supports=False, quote_check_passed=False: 1" in out
    assert "total: 5" in out
    assert "gap rate: 66.7%" in out
    assert "langfuse:ft1" in out
    assert "langfuse:ft2" in out


def test_report_gap_rate_na_when_no_quote_passed(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {"supports": True, "quote_check_passed": False},
        {"supports": False, "quote_check_passed": False},
    ]
    (files / "claim_support.json").write_text(json.dumps(rows), encoding="utf-8")
    lec.report()
    assert "gap rate: n/a (no quote_check_passed=True rows)" in capsys.readouterr().out


def test_label_append_only(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_queue(files)
    (files / "claim_support.json").write_text(
        json.dumps([{"seed": True}]), encoding="utf-8"
    )
    patch_pipeline(monkeypatch)
    lec.label("t1", "Prakash", None, ["0:supports"])
    rows = read(files / "claim_support.json")
    assert rows[0] == {"seed": True}
    assert rows[1]["claim"] == "c0"


def test_label_requires_by(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_queue(files)
    with pytest.raises(SystemExit):
        lec.label("t1", "", "u1", None)


def test_label_all_skip_without_uris_is_not_labeled(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    monkeypatch.setattr(
        lec, "retrieve_postgres", lambda *a: pytest.fail("pipeline called")
    )
    with pytest.raises(SystemExit):
        lec.label("t1", "Prakash", None, ["0:skip"])
    assert not (files / "labeled_traffic.json").exists()
    assert not (files / "claim_support.json").exists()
    assert read(files / "_traffic_queue.json")[0]["status"] == "pending"


def test_skip_writes_no_goldens_and_leaves_pending_list(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_queue(files)
    lec.skip("t1", "Prakash", "bad")
    assert not (files / "labeled_traffic.json").exists()
    assert not (files / "claim_support.json").exists()
    lec.list_candidates("pending")
    assert capsys.readouterr().out.strip().endswith("skipped")


def test_gate_verdict_abstained_comes_from_validator(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_queue(files)
    claims = [
        {"claim": "c0", "quote": "q0", "evidence_id": "u0"},
        {"claim": "c1", "quote": "q1", "evidence_id": "u1"},
    ]
    patch_pipeline(monkeypatch, claims)
    lec.label("t1", "Prakash", None, ["1:supports"])
    assert read(files / "claim_support.json")[0]["gate_verdict_abstained"] is False


def test_fetch_refuses_non_string_trace_input(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lec, "get_lf_client", lambda: Client([trace("bad", {"q": "x"})])
    )
    with pytest.raises(SystemExit):
        lec.fetch(7, 50)


def test_fetch_requires_langfuse_client(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lec, "get_lf_client", lambda: None)
    with pytest.raises(SystemExit):
        lec.fetch(7, 50)
