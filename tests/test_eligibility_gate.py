from __future__ import annotations

import inspect
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

import psycopg2
import pytest

import app.retrieval.eligibility_gate as eg
from app.retrieval.eligibility_gate import eligible_chunk_ids


class FilteringCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.input_rows = rows
        self.rows: list[tuple[str]] = []
        self.sql = ""
        self.params: dict[str, Any] = {}
        self.result: tuple[bool] | None = None

    def __enter__(self) -> "FilteringCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any] | tuple[Any, ...]) -> None:
        self.sql = sql
        if isinstance(params, tuple):
            component_uri, as_of = params
            self.result = (self._component_eligible(str(component_uri), as_of),)
            return

        self.params = params
        as_of = params["as_of"]
        self.rows = [
            (row["id"],)
            for row in self.input_rows
            if row["ingestion_status"] == "approved"
            and row["source_type"] != "nkp_case"
            and (
                (
                    row.get("component_uri") is not None
                    and self._component_eligible(str(row["component_uri"]), as_of)
                )
                or (
                    row.get("component_uri") is None
                    and row.get("effective_date_ad") is not None
                    and row["effective_date_ad"] <= as_of
                )
            )
            and (
                row.get("component_uri") is None or self._expression_current(row, as_of)
            )
        ]

    def fetchall(self) -> list[tuple[str]]:
        return self.rows

    def fetchone(self) -> tuple[bool] | None:
        return self.result

    def _component_eligible(self, component_uri: str, as_of: date) -> bool:
        effects = [
            effect
            for row in self.input_rows
            if row.get("component_uri") == component_uri
            for effect in row.get("effects", [])
        ]
        commenced = any(
            effect["effect_type"] == "commence"
            and effect.get("approval_status", "approved") == "approved"
            and effect["start"] <= as_of
            and (effect.get("end") is None or as_of < effect["end"])
            and effect.get("commencement_dependency") is None
            for effect in effects
        )
        terminated = any(
            effect["effect_type"] in {"repeal", "expiry", "declared_invalid", "suspend"}
            and effect.get("approval_status", "approved") == "approved"
            and effect["start"] <= as_of
            for effect in effects
        )
        return commenced and not terminated

    def _expression_current(self, row: dict[str, Any], as_of: date) -> bool:
        return not any(
            effect["effect_type"] == "amend"
            and effect.get("approval_status", "approved") == "approved"
            and effect.get("valid_lower") is not None
            and effect["valid_lower"] <= as_of
            and effect["transaction_start"] > row["created_at"]
            for effect in row.get("effects", [])
        )


class FilteringConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.cursor_obj = FilteringCursor(rows)

    def cursor(self) -> FilteringCursor:
        return self.cursor_obj


def _row(**extra: Any) -> dict[str, Any]:
    return {
        "id": "chunk-1",
        "ingestion_status": "approved",
        "source_type": "act",
        "effective_date_ad": date(2020, 1, 1),
        "component_uri": "/law/1",
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "effects": [
            {"effect_type": "commence", "start": date(2020, 1, 1)},
        ],
        **extra,
    }


def test_eligible_chunk_ids_sql_does_not_read_llm_metadata() -> None:
    source = inspect.getsource(eg.eligible_chunk_ids)
    assert not any(
        column in source for column in ("summary", "keywords", "relevant_questions")
    )


def test_eligible_chunk_ids_keeps_approved_non_case_checks() -> None:
    conn = FilteringConn(
        [
            _row(id="pending", ingestion_status="pending"),
            _row(id="case", source_type="nkp_case"),
            _row(id="law"),
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == {"law"}


def test_eligible_chunk_ids_excludes_declared_invalid_pre_retrieval() -> None:
    conn = FilteringConn(
        [
            _row(
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {"effect_type": "declared_invalid", "start": date(2023, 1, 1)},
                ]
            )
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()


def test_eligible_chunk_ids_excludes_suspended_pre_retrieval() -> None:
    conn = FilteringConn(
        [
            _row(
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {"effect_type": "suspend", "start": date(2023, 1, 1)},
                ]
            )
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()


def test_eligible_chunk_ids_excludes_not_yet_commenced_pre_retrieval() -> None:
    conn = FilteringConn(
        [_row(effects=[{"effect_type": "commence", "start": date(2025, 1, 1)}])]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()


def test_eligible_chunk_ids_excludes_pending_commencement_dependency() -> None:
    conn = FilteringConn(
        [
            _row(
                effects=[
                    {
                        "effect_type": "commence",
                        "start": date(2020, 1, 1),
                        "commencement_dependency": "gazette_notification",
                    }
                ]
            )
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == set()


def test_eligible_chunk_ids_keeps_unlinked_effective_date_fallback() -> None:
    conn = FilteringConn(
        [
            _row(id="old", component_uri=None, effective_date_ad=date(2020, 1, 1)),
            _row(id="future", component_uri=None, effective_date_ad=date(2025, 1, 1)),
            _row(id="null", component_uri=None, effective_date_ad=None),
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 1, 1)) == {"old"}


def test_eligible_chunk_ids_excludes_stale_expression_pre_retrieval() -> None:
    conn = FilteringConn(
        [
            _row(
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {
                        "effect_type": "amend",
                        "valid_lower": date(2023, 1, 1),
                        "transaction_start": datetime(2024, 2, 1, tzinfo=timezone.utc),
                    },
                ]
            )
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 3, 1)) == set()


def test_eligible_chunk_ids_keeps_refreshed_expression_after_amend_known() -> None:
    conn = FilteringConn(
        [
            _row(
                created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {
                        "effect_type": "amend",
                        "valid_lower": date(2023, 1, 1),
                        "transaction_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    },
                ],
            )
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 3, 1)) == {"chunk-1"}


def test_eligible_chunk_ids_keeps_pending_or_unresolved_amend() -> None:
    conn = FilteringConn(
        [
            _row(
                id="pending",
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {
                        "effect_type": "amend",
                        "approval_status": "pending",
                        "valid_lower": date(2023, 1, 1),
                        "transaction_start": datetime(2024, 2, 1, tzinfo=timezone.utc),
                    },
                ],
            ),
            _row(
                id="unresolved",
                effects=[
                    {"effect_type": "commence", "start": date(2020, 1, 1)},
                    {
                        "effect_type": "amend",
                        "valid_lower": None,
                        "transaction_start": datetime(2024, 2, 1, tzinfo=timezone.utc),
                    },
                ],
            ),
        ]
    )
    assert eligible_chunk_ids(cast(Any, conn), date(2024, 3, 1)) == {
        "pending",
        "unresolved",
    }


def test_is_expression_current_migration_uses_transaction_time_and_empty_safe() -> None:
    sql = Path("migrations/012_expression_staleness_gate.sql").read_text()
    assert "CREATE OR REPLACE FUNCTION is_expression_current" in sql
    assert "lower(transaction_time) > p_chunk_created_at" in sql
    assert "lower(legal_valid_time) <= p_as_of::timestamptz" in sql
    assert "effective_date" not in sql


@pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="SUPABASE_DB_URL not set")
def test_is_expression_current_live_db_cases() -> None:
    component = "/test/expression-staleness"
    with psycopg2.connect(os.environ["SUPABASE_DB_URL"]) as conn, conn.cursor() as cur:
        cur.execute("SAVEPOINT expression_staleness")
        cur.execute("DELETE FROM lifecycle_effect WHERE component_uri=%s", (component,))
        cur.execute(
            """
            INSERT INTO lifecycle_effect
                (component_uri, effect_type, approval_status, legal_valid_time, transaction_time)
            VALUES
                (%(component)s, 'amend', 'approved', '[2023-01-01,)'::tstzrange,
                 '[2024-02-01,)'::tstzrange),
                (%(component)s, 'amend', 'pending', '[2023-01-01,)'::tstzrange,
                 '[2024-03-01,)'::tstzrange),
                (%(component)s, 'amend', 'approved', 'empty'::tstzrange,
                 '[2024-03-01,)'::tstzrange)
            """,
            {"component": component},
        )
        cur.execute(
            "SELECT is_expression_current(%s, %s, %s)",
            (component, datetime(2024, 1, 1, tzinfo=timezone.utc), date(2024, 3, 1)),
        )
        assert cur.fetchone() == (False,)
        cur.execute(
            "SELECT is_expression_current(%s, %s, %s)",
            (component, datetime(2024, 2, 2, tzinfo=timezone.utc), date(2024, 3, 1)),
        )
        assert cur.fetchone() == (True,)
        cur.execute("DELETE FROM lifecycle_effect WHERE component_uri=%s", (component,))
        cur.execute(
            """
            INSERT INTO lifecycle_effect
                (component_uri, effect_type, approval_status, legal_valid_time, transaction_time)
            VALUES
                (%(component)s, 'amend', 'pending', '[2023-01-01,)'::tstzrange,
                 '[2024-03-01,)'::tstzrange),
                (%(component)s, 'amend', 'approved', 'empty'::tstzrange,
                 '[2024-03-01,)'::tstzrange)
            """,
            {"component": component},
        )
        cur.execute(
            "SELECT is_expression_current(%s, %s, %s)",
            (component, datetime(2024, 1, 1, tzinfo=timezone.utc), date(2024, 3, 1)),
        )
        assert cur.fetchone() == (True,)
        cur.execute("ROLLBACK TO SAVEPOINT expression_staleness")


def test_eligible_compatibility_calls_canonical_sql() -> None:
    conn = FilteringConn([_row(component_uri="/law/1")])
    assert eg.is_eligible(cast(Any, conn), "/law/1", date(2024, 1, 1))
    assert conn.cursor_obj.sql == "SELECT is_eligible(%s, %s)"
