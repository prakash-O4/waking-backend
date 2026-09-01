# mypy: ignore-errors
from __future__ import annotations

from scripts import review_documents as rd

A = "alice"
B = "bob"


class Conn:
    def __init__(self):
        self.rows = {}
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
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=()):
        sql1 = " ".join(sql.split())
        if sql1.startswith("SELECT ingestion_status, approved_by"):
            row = self.conn.rows.get(params[0])
            self.result = None if row is None else (row["status"], row["a1"], row["a2"])
        elif sql1.startswith("SELECT ingestion_status FROM documents"):
            row = self.conn.rows.get(params[0])
            self.result = None if row is None else (row["status"],)
        elif sql1.startswith("UPDATE documents SET approved_by"):
            self.conn.rows[params[1]]["a1"] = params[0]
        elif "SET second_approved_by" in sql1:
            self.conn.rows[params[1]]["a2"] = params[0]
            self.conn.rows[params[1]]["status"] = "approved"
        elif "SET ingestion_status='rejected'" in sql1:
            self.conn.rows[params[1]]["status"] = "rejected"
            self.conn.rows[params[1]]["a1"] = params[0]
        elif sql1.startswith("SELECT id, source_type"):
            self.result = [
                (
                    k,
                    v["source_type"],
                    v["source_id"],
                    v["ingested_at"],
                    v["redaction_failed"],
                )
                for k, v in self.conn.rows.items()
                if v["status"] == "pending"
            ]
        else:
            raise AssertionError(sql1)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result or []


def conn_with(row_status="pending", a1=None):
    conn = Conn()
    conn.rows["d1"] = {
        "status": row_status,
        "a1": a1,
        "a2": None,
        "source_type": "act",
        "source_id": "law-1",
        "ingested_at": "now",
        "redaction_failed": False,
    }
    return conn


def test_first_approve_stays_pending():
    conn = conn_with()
    assert rd.approve_one(conn, "d1", A)
    assert conn.rows["d1"]["a1"] == A
    assert conn.rows["d1"]["status"] == "pending"


def test_second_approve_different_person_approves():
    conn = conn_with(a1=A)
    assert rd.approve_one(conn, "d1", B)
    assert conn.rows["d1"]["a2"] == B
    assert conn.rows["d1"]["status"] == "approved"


def test_second_approve_same_person_refused():
    conn = conn_with(a1=A)
    assert not rd.approve_one(conn, "d1", A)
    assert conn.rows["d1"]["status"] == "pending"
    assert conn.rollbacks == 1


def test_approve_already_done_refused():
    for status in ("approved", "rejected"):
        conn = conn_with(row_status=status)
        assert not rd.approve_one(conn, "d1", A)
        assert conn.rows["d1"]["status"] == status


def test_reject_pending_records_reviewer():
    conn = conn_with()
    assert rd.reject_one(conn, "d1", A)
    assert conn.rows["d1"]["status"] == "rejected"
    assert conn.rows["d1"]["a1"] == A


def test_list_pending_documents(capsys):
    conn = conn_with()
    rd.list_pending(conn)
    out = capsys.readouterr().out
    assert "d1" in out
    assert "law-1" in out
