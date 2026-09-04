from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

_BM25_MARKER = "-- pg_search (ParadeDB)"

# Ordered list of (name, path).
# 005 is split into core + bm25; paths are resolved dynamically.
MIGRATIONS: list[tuple[str, str | None]] = [
    ("001_bitemporal_schema", "001_bitemporal_schema.sql"),
    ("002_gate_suspend_fix", "002_gate_suspend_fix.sql"),
    ("003_bs_ad_calendar", "003_bs_ad_calendar.sql"),
    ("004_precedent_schema", "004_precedent_schema.sql"),
    ("005_ingestion_pipeline_core", None),  # computed below
    ("005_ingestion_pipeline_bm25", None),  # optional; computed below
    ("006_add_summary", "006_add_summary.sql"),
    ("007_retrieval_indexes", "007_retrieval_indexes.sql"),
    ("008_work_relations", "008_work_relations.sql"),
    ("009_lifecycle_raw_clause", "009_lifecycle_raw_clause.sql"),
    ("010_metadata_provenance_comments", "010_metadata_provenance_comments.sql"),
    ("011_chunk_authority_links", "011_chunk_authority_links.sql"),
    ("012_expression_staleness_gate", "012_expression_staleness_gate.sql"),
]


def _split_005() -> tuple[str, str]:
    sql = (MIGRATIONS_DIR / "005_ingestion_pipeline.sql").read_text()
    idx = sql.find(_BM25_MARKER)
    return (sql[:idx].strip(), sql[idx:].strip()) if idx != -1 else (sql, "")


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def _ensure_migrations_table(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name        TEXT PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def _seed_existing(cur: Any) -> None:
    """
    If the DB was migrated before schema_migrations existed, detect the
    already-applied migrations by checking for their sentinel objects and
    insert them so subsequent runs skip them correctly.
    """
    sentinels = {
        "001_bitemporal_schema": "SELECT 1 FROM pg_tables WHERE tablename = 'work'",
        "002_gate_suspend_fix": "SELECT 1 FROM pg_tables WHERE tablename = 'work'",  # same table; 002 just alters
        "003_bs_ad_calendar": "SELECT 1 FROM pg_tables WHERE tablename = 'bs_ad_calendar'",
        "004_precedent_schema": "SELECT 1 FROM pg_tables WHERE tablename = 'precedent'",
        "005_ingestion_pipeline_core": "SELECT 1 FROM pg_tables WHERE tablename = 'chunks'",
        "005_ingestion_pipeline_bm25": "SELECT 1 FROM pg_extension WHERE extname = 'pg_search'",
        "006_add_summary": "SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='summary'",
        "007_retrieval_indexes": "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_chunk_text_fts'",
        "008_work_relations": "SELECT 1 FROM pg_tables WHERE tablename = 'work_relations'",
        "009_lifecycle_raw_clause": "SELECT 1 FROM information_schema.columns WHERE table_name='lifecycle_effect' AND column_name='raw_clause_text'",
        "011_chunk_authority_links": "SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='component_uri'",
        "012_expression_staleness_gate": "SELECT 1 FROM pg_proc WHERE proname = 'is_expression_current'",
    }
    for name, probe in sentinels.items():
        cur.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (name,))
        if cur.fetchone():
            continue  # already tracked
        cur.execute(probe)
        if cur.fetchone():
            cur.execute(
                "INSERT INTO schema_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
                (name,),
            )
            print(f"  {name}: detected as already applied, registered")


def _applied(cur: Any, name: str) -> bool:
    cur.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (name,))
    return cur.fetchone() is not None


def _mark(cur: Any, name: str) -> None:
    cur.execute(
        "INSERT INTO schema_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
        (name,),
    )


def main() -> None:
    url = _db_url()
    conn = psycopg2.connect(url)
    conn.autocommit = False

    core_005, bm25_005 = _split_005()

    sql_map: dict[str, str] = {
        "005_ingestion_pipeline_core": core_005,
        "005_ingestion_pipeline_bm25": bm25_005,
    }
    for name, filename in MIGRATIONS:
        if filename:
            sql_map[name] = (MIGRATIONS_DIR / filename).read_text()

    try:
        with conn.cursor() as cur:
            _ensure_migrations_table(cur)
            _seed_existing(cur)
        conn.commit()

        for name, _ in MIGRATIONS:
            sql = sql_map.get(name, "")
            if not sql.strip():
                continue

            with conn.cursor() as cur:
                if _applied(cur, name):
                    print(f"  {name}: already applied, skipping")
                    continue

            label = name + (" (optional)" if "bm25" in name else "")
            print(f"  applying {label} ...", end=" ", flush=True)
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    _mark(cur, name)
                conn.commit()
                print("ok")
            except Exception as exc:
                conn.rollback()
                if "bm25" in name:
                    print(f"skipped ({exc.__class__.__name__}: ParadeDB not available)")
                    with conn.cursor() as cur:
                        _mark(cur, name)
                    conn.commit()
                else:
                    raise

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("done.")


if __name__ == "__main__":
    main()
