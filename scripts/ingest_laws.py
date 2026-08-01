from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.authority.parser import parse_law  # noqa: E402
from app.authority.writer import (  # noqa: E402
    connect,
    insert_commence,
    upsert_component,
    upsert_expression,
    upsert_source,
    upsert_work,
)
from app.search.client import DEFAULT_INDEX, ensure_index, get_client  # noqa: E402

FALLBACK_AS_OF = date(1900, 1, 1)


def _index_doc(
    os_client: Any,
    component_uri: str,
    as_of: date,
    text_ne: str,
    text_hash: str,
    work_title_ne: str,
) -> None:
    os_client.index(
        index=DEFAULT_INDEX,
        id=component_uri,
        body={
            "component_uri": component_uri,
            "as_of": as_of.isoformat(),
            "text_ne": text_ne,
            "text_hash": text_hash,
            "work_title_ne": work_title_ne,
            "dense_vector": [0.0] * 1536,
        },
        refresh=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    ensure_index()
    os_client = get_client()
    path = ROOT / "laws.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()[args.offset :]
    if args.limit is not None:
        lines = lines[: args.limit]

    laws_done = components_done = 0
    with connect() as conn:
        for line_no, line in enumerate(lines, start=args.offset + 1):
            try:
                record = json.loads(line)
                law = parse_law(record)
                if law.enactment_ad is None:
                    print(
                        f"warning: no enactment date for {law.uri}; lifecycle skipped"
                    )
                work_id = upsert_work(conn, law)
                source_id = upsert_source(conn, work_id, law, record.get("url"))
                for component in law.components:
                    upsert_component(conn, work_id, component)
                    as_of = law.enactment_ad or FALLBACK_AS_OF
                    if law.enactment_ad is not None:
                        insert_commence(
                            conn, component.uri, source_id, law.enactment_ad
                        )
                    upsert_expression(conn, component, as_of)
                    _index_doc(
                        os_client,
                        component.uri,
                        as_of,
                        component.text_ne,
                        component.text_hash,
                        law.title_ne,
                    )
                    components_done += 1
                    if components_done % 100 == 0:
                        conn.commit()
                laws_done += 1
                if laws_done % 50 == 0:
                    conn.commit()
                    print(f"ingested {laws_done} laws ({components_done} components)")
            except Exception as exc:  # noqa: BLE001 - keep corpus batch moving.
                conn.rollback()
                print(f"error: law line {line_no} failed: {exc}")
        conn.commit()
    os_client.indices.refresh(index=DEFAULT_INDEX)
    print(f"done: {laws_done} laws, {components_done} components")


if __name__ == "__main__":
    main()
