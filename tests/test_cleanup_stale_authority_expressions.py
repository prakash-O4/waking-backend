# mypy: ignore-errors
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

from scripts.cleanup_stale_authority_expressions import _cleanup_orphan_components


class FakeConn:
    def __init__(self) -> None:
        self.component_uris = {"/law/dafa/1", "/law/dafa/2", "/law/dafa/3"}
        self.expressions = Counter({"/law/dafa/2": 2, "/law/dafa/3": 1})
        self.lifecycle = {
            "/law/dafa/2": Counter({"pending": 1}),
            "/law/dafa/3": Counter({"approved": 1}),
        }

    def cursor(self) -> Any:
        return FakeCursor(self)


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.result: list[tuple[Any, ...]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        squashed = " ".join(sql.split())
        uri = str(params[0]) if params else ""
        if squashed.startswith("SELECT uri FROM component"):
            self.result = [(uri,) for uri in sorted(self.conn.component_uris)]
        elif squashed.startswith("SELECT approval_status"):
            self.result = list(self.conn.lifecycle.get(uri, Counter()).items())
        elif squashed.startswith("SELECT COUNT(*) FROM expression"):
            self.result = [(self.conn.expressions[uri],)]
        elif squashed.startswith("DELETE FROM expression"):
            self.conn.expressions[uri] = 0
        elif squashed.startswith("DELETE FROM lifecycle_effect"):
            self.conn.lifecycle.pop(uri, None)
        elif squashed.startswith("DELETE FROM component"):
            self.conn.component_uris.discard(uri)
        else:
            raise AssertionError(squashed)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result[0] if self.result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.result


def law() -> SimpleNamespace:
    return SimpleNamespace(components=[SimpleNamespace(uri="/law/dafa/1")])


def test_orphan_cleanup_counts_and_blocks_approved() -> None:
    conn = FakeConn()

    counts = _cleanup_orphan_components(conn, "work1", law(), dry_run=False)

    assert counts["orphan_components"] == 1
    assert counts["orphan_expressions"] == 2
    assert counts["orphan_lifecycle_effects"] == 1
    assert counts["orphan_components_blocked"] == 1
    assert "/law/dafa/2" not in conn.component_uris
    assert "/law/dafa/3" in conn.component_uris


def test_orphan_cleanup_dry_run_does_not_delete() -> None:
    conn = FakeConn()

    counts = _cleanup_orphan_components(conn, "work1", law(), dry_run=True)

    assert counts["orphan_components"] == 1
    assert counts["orphan_components_blocked"] == 1
    assert conn.component_uris == {"/law/dafa/1", "/law/dafa/2", "/law/dafa/3"}
