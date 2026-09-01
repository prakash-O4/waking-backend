# mypy: ignore-errors
from __future__ import annotations

from app.ingestion.pipeline import _content_hash
from scripts import recompute_content_hashes as rch


class Conn:
    def __init__(self):
        self.docs = [("d1", "src1", "old"), ("d2", "src2", _content_hash("same"))]
        self.hashes = {"d1": "old", "d2": _content_hash("same")}
        self.status = {"d1": "approved", "d2": "approved"}
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=()):
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT id, source_id, content_hash"):
            self.result = self.conn.docs
        elif squashed.startswith("UPDATE documents SET content_hash"):
            new_hash, doc_id = params
            self.conn.hashes[doc_id] = new_hash
        else:
            raise AssertionError(squashed)

    def fetchall(self):
        return self.result


def test_dry_run_does_not_update_or_commit():
    conn = Conn()
    counts = rch.run(
        conn, {"src1": {"content": "दफा १"}, "src2": {"content": "same"}}, dry_run=True
    )
    assert counts["would_update"] == 1
    assert conn.hashes["d1"] == "old"
    assert conn.commits == 0


def test_recompute_updates_hash_only_not_status():
    conn = Conn()
    counts = rch.run(conn, {"src1": {"content": "दफा १"}, "src2": {"content": "same"}})
    assert counts["updated"] == 1
    assert counts["unchanged"] == 1
    assert conn.hashes["d1"] == _content_hash("दफा १")
    assert conn.status["d1"] == "approved"
    assert conn.commits == 1


def test_missing_record_skips():
    conn = Conn()
    counts = rch.run(conn, {})
    assert counts["missing_record"] == 2
    assert conn.commits == 0
