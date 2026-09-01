# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from app.ingestion.pipeline import _content_hash
from scripts import backfill_authority_layer as bal


@dataclass
class FakeComponent:
    uri: str
    component_type: str = "dafa"
    number: str = "1"
    text_ne: str = "text"
    text_hash: str = "hash"


class FakeConn:
    def __init__(self) -> None:
        self.docs = [("doc1", "src1", "act", _content_hash("content"))]
        self.doc_work = {"doc1": "work1"}
        self.work_uris = {"work1": "/law/1"}
        self.commits = 0
        self.rollbacks = 0
        self.inserted: set[tuple[str, str]] = set()
        self._snapshot: set[tuple[str, str]] | None = None

    def ensure_txn(self) -> None:
        if self._snapshot is None:
            self._snapshot = set(self.inserted)

    def cursor(self) -> Any:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self._snapshot = None

    def rollback(self) -> None:
        self.rollbacks += 1
        if self._snapshot is not None:
            self.inserted = self._snapshot
            self._snapshot = None


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.result: list[tuple[Any, ...]] = []
        self.rowcount = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        squashed = " ".join(sql.split())
        self.rowcount = 0
        if squashed.startswith("SELECT id, source_id"):
            self.result = self.conn.docs
        elif squashed.startswith("SELECT DISTINCT work_id"):
            work_id = self.conn.doc_work.get(str(params[0]))
            self.result = [(work_id,)] if work_id else []
        elif squashed.startswith("SELECT uri FROM work"):
            uri = self.conn.work_uris.get(str(params[0]))
            self.result = [(uri,)] if uri else []
        elif squashed.startswith("SELECT COUNT(*)"):
            table = squashed.split(" FROM ", 1)[1]
            self.result = [(sum(1 for kind, _ in self.conn.inserted if kind == table),)]
        elif squashed.startswith("SELECT commencement_dependency"):
            count = sum(
                1 for kind, _ in self.conn.inserted if kind == "lifecycle_effect"
            )
            self.result = [(None, "यो ऐन तुरुन्त प्रारम्भ हुनेछ", count)] if count else []
        elif squashed.startswith("UPDATE documents SET source_pub_id"):
            self.conn.ensure_txn()
            self.conn.inserted.add(("document_source_link", str(params[0])))
            self.rowcount = 1
        elif squashed.startswith("UPDATE chunks SET component_uri"):
            self.conn.ensure_txn()
            self.conn.inserted.add(("chunk_component_link", str(params[0])))
            self.rowcount = 1
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result[0] if self.result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.result


def record(content: str = "content") -> dict[str, Any]:
    return {"_id": "src1", "name": "x", "document_type": "act", "content": content}


def parsed_law(uri: str = "/law/1", n: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        uri=uri,
        source_sha256="sha",
        components=[FakeComponent(f"{uri}/dafa/{i}", number=str(i)) for i in range(n)],
    )


def patch_writers(monkeypatch, calls: list[tuple[str, Any]]) -> None:
    def source(conn: FakeConn, work_id: str, law: Any, source_url: None = None) -> str:
        calls.append(("source", work_id))
        conn.ensure_txn()
        conn.inserted.add(("source_publication", work_id))
        return "source1"

    def component(conn: FakeConn, work_id: str, comp: Any) -> None:
        calls.append(("component", comp.uri))
        conn.ensure_txn()
        conn.inserted.add(("component", comp.uri))

    def expression(conn: FakeConn, comp: Any, as_of: Any) -> None:
        calls.append(("expression", comp.uri))
        conn.ensure_txn()
        conn.inserted.add(("expression", comp.uri))

    def commence(*, law: Any, content: str, source_pub_id: str, conn: FakeConn) -> None:
        calls.append(("commence", source_pub_id))
        conn.ensure_txn()
        conn.inserted.add(("lifecycle_effect", law.uri))

    monkeypatch.setattr(bal, "upsert_source", source)
    monkeypatch.setattr(bal, "upsert_component", component)
    monkeypatch.setattr(bal, "upsert_expression", expression)
    monkeypatch.setattr(bal, "extract_commencement_proposals", commence)


def test_happy_path_calls_all_writers(monkeypatch) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(bal, "parse_law", lambda _record: parsed_law())
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {"src1": record()})

    assert summary.counts["processed"] == 1
    assert [c[0] for c in calls] == [
        "source",
        "component",
        "expression",
        "component",
        "expression",
        "commence",
    ]
    assert summary.counts["chunk_links_written"] == 2
    assert conn.commits == 1


def test_missing_source_id_skips(monkeypatch) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {})

    assert summary.counts["missing_record"] == 1
    assert calls == []


def test_hash_mismatch_skips(monkeypatch) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {"src1": record("changed")})

    assert summary.counts["hash_mismatch"] == 1
    assert calls == []


def test_missing_work_id_skips(monkeypatch) -> None:
    conn = FakeConn()
    conn.doc_work = {}
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(bal, "parse_law", lambda _record: parsed_law())
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {"src1": record()})

    assert summary.counts["missing_work_id"] == 1
    assert calls == []


def test_uri_mismatch_skips_and_warns(monkeypatch, capsys) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(bal, "parse_law", lambda _record: parsed_law("/other"))
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {"src1": record()})

    assert summary.counts["uri_mismatch"] == 1
    assert calls == []
    assert "law.uri != work.uri" in capsys.readouterr().err


def test_second_run_adds_no_new_rows(monkeypatch) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(bal, "parse_law", lambda _record: parsed_law())
    patch_writers(monkeypatch, calls)

    first = bal.run_backfill(conn, {"src1": record()})
    second = bal.run_backfill(conn, {"src1": record()})

    assert first.counts["components_written"] == 2
    assert second.counts["components_written"] == 0
    assert second.counts["sources_written"] == 0
    assert second.counts["expressions_written"] == 0
    assert second.counts["lifecycle_written"] == 0


def test_dry_run_checks_but_does_not_write_or_commit(monkeypatch) -> None:
    conn = FakeConn()
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(bal, "parse_law", lambda _record: parsed_law())
    patch_writers(monkeypatch, calls)

    summary = bal.run_backfill(conn, {"src1": record()}, dry_run=True)

    assert summary.counts["processed"] == 1
    assert summary.counts["components_seen"] == 2
    assert calls == []
    assert conn.commits == 0


def test_mid_write_failure_rolls_back_and_continues(monkeypatch) -> None:
    conn = FakeConn()
    conn.docs.append(("doc2", "src2", "act", _content_hash("content2")))
    conn.doc_work["doc2"] = "work2"
    conn.work_uris["work2"] = "/law/2"

    monkeypatch.setattr(
        bal,
        "parse_law",
        lambda rec: parsed_law("/law/2" if rec["_id"] == "src2" else "/law/1"),
    )
    calls: list[tuple[str, Any]] = []
    patch_writers(monkeypatch, calls)

    def flaky_expression(conn: FakeConn, comp: Any, as_of: Any) -> None:
        conn.ensure_txn()
        if comp.uri == "/law/1/dafa/1":
            raise RuntimeError("boom")
        conn.inserted.add(("expression", comp.uri))

    monkeypatch.setattr(bal, "upsert_expression", flaky_expression)

    summary = bal.run_backfill(
        conn, {"src1": record(), "src2": {**record("content2"), "_id": "src2"}}
    )

    assert summary.counts["write_exception"] == 1
    assert summary.counts["processed"] == 1
    assert ("component", "/law/1/dafa/0") not in conn.inserted
    assert conn.rollbacks >= 1
    assert conn.commits == 1
