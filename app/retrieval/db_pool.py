from __future__ import annotations

import psycopg2.pool

from app.authority.writer import database_url


def make_retrieval_pool(maxconn: int) -> psycopg2.pool.ThreadedConnectionPool:
    return psycopg2.pool.ThreadedConnectionPool(1, maxconn, database_url())
