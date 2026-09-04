from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Any, cast


TEXT = "दफा १ अनुसार करदाताले आयकर स्वीकार गर्नुपर्ने हुन्छ र म्यादभित्र बुझाउनुपर्छ।"
QUOTE = "करदाताले आयकर स्वीकार गर्नुपर्ने हुन्छ"


def sha(text: str = TEXT) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def effect(kind: str, start: date, **extra: Any) -> dict[str, Any]:
    return {"effect_type": kind, "start": start, "approval_status": "approved", **extra}


class StressCursor:
    def __init__(self, conn: "StressConn") -> None:
        self.conn = conn
        self.result: tuple[Any, ...] | None = None
        self.rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> "StressCursor":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: dict[str, Any] | tuple[Any, ...]) -> None:
        s = " ".join(sql.split())
        self.result = None
        self.rows = []
        if s.startswith("SELECT is_eligible"):
            component_uri, as_of = params
            self.result = (
                self.conn.component_eligible(str(component_uri), cast(date, as_of)),
            )
        elif s.startswith("SELECT c.id::text FROM chunks c JOIN documents d"):
            as_of = cast(dict[str, Any], params)["as_of"]
            self.rows = [
                (cid,)
                for cid, chunk in self.conn.chunks.items()
                if chunk.get("ingestion_status", "approved") == "approved"
                and chunk.get("source_type", "act") != "nkp_case"
                and self.conn.chunk_eligible(cid, as_of)
            ]
        elif s.startswith("SELECT chunk_text, span_sha256 FROM chunks"):
            chunk = self.conn.chunks.get(
                str(cast(dict[str, Any], params)["component_uri"])
            )
            self.result = (chunk["text"], chunk["hash"]) if chunk else None
        elif s.startswith("SELECT component_uri FROM chunks"):
            chunk = self.conn.chunks.get(
                str(cast(dict[str, Any], params)["evidence_id"])
            )
            self.result = (chunk.get("component_uri"),) if chunk else None
        elif s.startswith("SELECT created_at FROM chunks"):
            chunk = self.conn.chunks.get(
                str(cast(dict[str, Any], params)["evidence_id"])
            )
            self.result = (chunk.get("created_at"),) if chunk else None
        elif s.startswith("SELECT is_expression_current"):
            component_uri, created_at, as_of = cast(tuple[Any, ...], params)
            self.result = (
                self.conn.expression_current(str(component_uri), created_at, as_of),
            )
        elif s.startswith("SELECT c.act_name"):
            chunk = self.conn.chunks.get(
                str(cast(dict[str, Any], params)["evidence_id"])
            )
            self.result = (
                (
                    chunk.get("act_name", "ऐन"),
                    None,
                    chunk.get("source_type", "act"),
                    "official_original",
                    0.99,
                    "https://example.test/source",
                )
                if chunk
                else None
            )
        elif s.startswith("SELECT w.title_ne"):
            self.result = ("परीक्षण ऐन", "Test Act")
        elif s.startswith("SELECT le.effective_date"):
            self.rows = []
        elif "JOIN chunks p ON p.id = c.co_retrieve_parent_id" in s:
            p = cast(dict[str, Any], params)
            child = self.conn.chunks.get(str(p["hit_id"]))
            parent_id = child.get("parent_id") if child else None
            parent = self.conn.chunks.get(str(parent_id)) if parent_id else None
            if parent and parent_id in set(p["eligible"]):
                self.result = (
                    p["hit_id"],
                    parent_id,
                    parent["text"],
                    parent["hash"],
                    parent.get("act_name", "Parent Act"),
                    None,
                    parent.get("chunk_type", "दफा"),
                    parent.get("section_number", ""),
                    parent.get("source_id", "src"),
                )
        elif "WHERE d.source_id = %(source_id)s" in s:
            p = cast(dict[str, Any], params)
            eligible = set(p["eligible"])
            for cid, chunk in self.conn.chunks.items():
                if (
                    cid in eligible
                    and chunk.get("source_id") == p["source_id"]
                    and chunk.get("section_number") == p["section_num"]
                ):
                    self.result = (
                        cid,
                        chunk["text"],
                        chunk["hash"],
                        chunk.get("act_name", "Act"),
                        None,
                        chunk.get("chunk_type", "दफा"),
                        chunk.get("section_number", ""),
                        chunk.get("source_id", ""),
                    )
                    break
        elif s == "SELECT work_id FROM chunks WHERE id = %s":
            chunk = self.conn.chunks.get(str(cast(tuple[Any, ...], params)[0]))
            self.result = (chunk.get("work_id"),) if chunk else None
        elif s.startswith("SELECT enabling_work_id"):
            rel = self.conn.relations.get(str(cast(dict[str, Any], params)["work_id"]))
            self.result = rel
        elif (
            s.startswith("SELECT c.id::text, c.chunk_text")
            and "c.work_id = %(work_id)s" in s
        ):
            p = cast(dict[str, Any], params)
            for cid, chunk in self.conn.chunks.items():
                if (
                    chunk.get("work_id") == p["work_id"]
                    and chunk.get("section_number") == p["section"]
                ):
                    self.result = (
                        cid,
                        chunk["text"],
                        chunk["hash"],
                        chunk.get("act_name", "Act"),
                        chunk.get("source_id", "src"),
                    )
                    break
        else:
            raise AssertionError(s)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class StressConn:
    def __init__(self, chunks: dict[str, dict[str, Any]]) -> None:
        self.chunks = chunks
        self.relations: dict[str, tuple[Any, ...]] = {}

    def cursor(self) -> StressCursor:
        return StressCursor(self)

    def chunk_eligible(self, chunk_id: str, as_of: date) -> bool:
        chunk = self.chunks[chunk_id]
        component_uri = chunk.get("component_uri")
        if component_uri is None:
            eff = chunk.get("effective_date_ad")
            return eff is not None and eff <= as_of
        return self.component_eligible(
            str(component_uri), as_of
        ) and self.expression_current(str(component_uri), chunk["created_at"], as_of)

    def component_eligible(self, component_uri: str, as_of: date) -> bool:
        effects = [
            eff
            for chunk in self.chunks.values()
            if chunk.get("component_uri") == component_uri
            for eff in chunk.get("effects", [])
        ]
        commenced = any(
            e["effect_type"] == "commence"
            and e.get("approval_status") == "approved"
            and e["start"] <= as_of
            and (e.get("end") is None or as_of < e["end"])
            and e.get("commencement_dependency") is None
            for e in effects
        )
        terminated = any(
            e["effect_type"] in {"repeal", "expiry", "declared_invalid", "suspend"}
            and e.get("approval_status") == "approved"
            and e["start"] <= as_of
            for e in effects
        )
        return commenced and not terminated

    def expression_current(
        self, component_uri: str, created_at: datetime, as_of: date
    ) -> bool:
        return not any(
            e["effect_type"] == "amend"
            and e.get("approval_status") == "approved"
            and e.get("start") is not None
            and e["start"] <= as_of
            and e.get("transaction_start", datetime.min.replace(tzinfo=timezone.utc))
            > created_at
            for chunk in self.chunks.values()
            if chunk.get("component_uri") == component_uri
            for e in chunk.get("effects", [])
        )


def chunk(
    component_uri: str, effects: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    return {
        "component_uri": component_uri,
        "text": TEXT,
        "hash": sha(),
        "source_type": "act",
        "ingestion_status": "approved",
        "act_name": "ऐन",
        "source_id": "src",
        "chunk_type": "दफा",
        "section_number": "१",
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "effects": effects,
        **extra,
    }
