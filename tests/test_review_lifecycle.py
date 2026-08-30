# mypy: ignore-errors
from __future__ import annotations

from psycopg2 import errors

from scripts import review_lifecycle as rl

A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"


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
        if sql1.startswith("SELECT approval_status, approved_by_1"):
            row = self.conn.rows.get(params[0])
            self.result = None if row is None else (row["status"], row["a1"], row["a2"])
        elif sql1.startswith("SELECT approval_status FROM lifecycle_effect"):
            row = self.conn.rows.get(params[0])
            self.result = None if row is None else (row["status"],)
        elif sql1.startswith("UPDATE lifecycle_effect SET approved_by_1"):
            self.conn.rows[params[1]]["a1"] = params[0]
        elif "SET approved_by_2" in sql1:
            self.conn.rows[params[1]]["a2"] = params[0]
            self.conn.rows[params[1]]["status"] = "approved"
        elif "SET approval_status='rejected'" in sql1:
            self.conn.rows[params[1]]["status"] = "rejected"
            self.conn.rows[params[1]]["a1"] = params[0]
        elif sql1.startswith("SELECT id FROM lifecycle_effect"):
            prefix = params[0][:-1]
            self.result = [
                (k,)
                for k, v in self.conn.rows.items()
                if v["component_uri"].startswith(prefix) and v["status"] == "pending"
            ]
        else:
            raise AssertionError(sql1)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result or []


def conn_with(row_status="pending", a1=None):
    conn = Conn()
    conn.rows["p1"] = {
        "status": row_status,
        "a1": a1,
        "a2": None,
        "component_uri": "/w/dafa/1",
    }
    return conn


def test_first_approve_stays_pending():
    conn = conn_with()
    assert rl.approve_one(conn, "p1", A)
    assert conn.rows["p1"]["a1"] == A
    assert conn.rows["p1"]["status"] == "pending"


def test_second_approve_different_uuid_approves():
    conn = conn_with(a1=A)
    assert rl.approve_one(conn, "p1", B)
    assert conn.rows["p1"]["a2"] == B
    assert conn.rows["p1"]["status"] == "approved"


def test_second_approve_same_uuid_refused():
    conn = conn_with(a1=A)
    assert not rl.approve_one(conn, "p1", A)
    assert conn.rows["p1"]["status"] == "pending"
    assert conn.rollbacks == 1


def test_approve_already_done_refused():
    for status in ("approved", "rejected"):
        conn = conn_with(row_status=status)
        assert not rl.approve_one(conn, "p1", A)
        assert conn.rows["p1"]["status"] == status


def test_reject_pending_records_rejecter():
    conn = conn_with()
    assert rl.reject_one(conn, "p1", A)
    assert conn.rows["p1"]["status"] == "rejected"
    assert conn.rows["p1"]["a1"] == A


def test_no_overlap_violation_is_caught(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise errors.ExclusionViolation()

    conn = conn_with(a1=A)
    monkeypatch.setattr(rl, "_approve_one", boom)
    assert not rl.approve_one(conn, "p1", B)
    assert "overlap" in capsys.readouterr().out
    assert conn.rollbacks == 1


def test_work_bulk_uses_same_row_logic():
    conn = Conn()
    conn.rows = {
        "p1": {
            "status": "pending",
            "a1": None,
            "a2": None,
            "component_uri": "/w/dafa/1",
        },
        "p2": {
            "status": "pending",
            "a1": None,
            "a2": None,
            "component_uri": "/w/dafa/2",
        },
    }
    assert rl.approve_work(conn, "/w", A)
    assert all(
        row["a1"] == A and row["status"] == "pending" for row in conn.rows.values()
    )
    assert rl.reject_work(conn, "/w", B)
    assert all(row["status"] == "rejected" for row in conn.rows.values())
