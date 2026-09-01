# mypy: ignore-errors
from __future__ import annotations

from scripts import backfill_source_kind as bsk


class Conn:
    def __init__(self, count=2):
        self.count = count
        self.updated = False
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
        squashed = " ".join(sql.split())
        if squashed.startswith("SELECT COUNT(*) FROM source_publication"):
            self.result = (self.conn.count,)
        elif squashed.startswith("UPDATE source_publication"):
            self.conn.updated = True
            self.conn.count = 0
        else:
            raise AssertionError(squashed)

    def fetchone(self):
        return self.result


def test_dry_run_reports_without_update():
    conn = Conn(count=3)
    assert bsk.run(conn, dry_run=True) == 3
    assert not conn.updated
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_real_run_updates_once_and_is_idempotent():
    conn = Conn(count=2)
    assert bsk.run(conn) == 2
    assert conn.updated
    assert conn.commits == 1

    conn.updated = False
    assert bsk.run(conn) == 0
    assert not conn.updated
