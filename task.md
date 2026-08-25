# Task: AGENT-3 — Authority Ranker + Cross-Reference Resolver (Stage 3)

**Branch:** `agent/stage-3-authority-ranker`
**Base:** `dev`
**Engineer:** Pi
**ADR:** `docs/adr-001-multi-agent-query-architecture.md` §Node 3 — Authority Ranker, §Node 4 — Cross-Reference Resolver

---

## Objective

Add two deterministic nodes after `retrieve_generate`:

1. **Authority Ranker** — sorts `all_hits` by `(tier ASC, rrf_score DESC)`. Tier is derived from `chunks.work_id → work.work_type`. Attaches `tier` and `conflict_flag` to each hit. No LLM call.
2. **Cross-Reference Resolver** — scans top-10 hits for Nepali section cross-references (`दफा X`, `उपदफा X`, `अनुसूची X`), fetches eligible co-referenced chunks, appends to `all_hits` with `co_retrieved: True`. No LLM call.

**Also included:** Remove dead `_REAL_MONOTONIC` variable from `gated_orchestrator.py` (flagged in PROGRESS.md).

New graph: `fact_extractor → retrieve_generate → authority_ranker → cross_ref_resolver → validate → assemble`

Behaviour of `validate_node` and `assemble_node` is unchanged. The enriched `all_hits` (sorted, with tier labels and co-retrieved chunks) will be consumed by Stage 4's Reasoner.

Stage 4 (Reasoner rewrite) is a separate task. Do not start it here.

---

## Acceptance criteria

1. `orchestrator.answer()` signature and return format unchanged.
2. All 7 existing tests in `tests/test_orchestrator.py` pass without modification — the two new nodes degrade gracefully when `conn = object()` has no `.cursor()` method (both functions have `except Exception` fallback).
3. `_authority_rank_hits(hits, conn)` — failure returns `hits` unchanged; success returns hits sorted by `(tier ASC, score DESC)` with `tier` and optional `conflict_flag` on each hit dict.
4. `_resolve_cross_refs(hits, as_of, conn)` — failure returns `[]`; success returns additional co-retrieved hits (not in original `all_hits`) with `co_retrieved: True`.
5. `_REAL_MONOTONIC` removed from `gated_orchestrator.py`.
6. 5 new tests added for the two new helper functions.
7. `make test` green (57 → 62 passing), `make lint` clean.
8. Zero-tolerance gates unaffected.

---

## Exact scope

**Modified files:**
- `app/retrieval/gated_orchestrator.py` — add imports (`re`, `eligible_chunk_ids`), remove `_REAL_MONOTONIC`, add 4 module-level constants, add `_authority_rank_hits()`, add `_resolve_cross_refs()`
- `app/retrieval/query_graph.py` — add `authority_ranker_node`, `cross_ref_resolver_node`, update graph edges
- `tests/test_orchestrator.py` — add 5 new tests

**No other files modified.** Do NOT touch `postgres_retriever.py`, `eligibility_gate.py`, `validation_gate.py`, `query_state.py`, or any eval file.

---

## Schema note — why `chunks.work_id → work.work_type`, not `documents.work_type`

The ADR says "tier from `work_type` on the `documents` table." That is incorrect — `documents` has no `work_type` column. The correct path is:

```sql
chunks.work_id → work.work_type   (NULL for nkp_case chunks, which have no work_id)
chunks.source_type                (fallback: 'nkp_case' → tier 6)
```

`WorkType` values from `app/authority/models.py`:
- `Constitution` → tier 1
- `Act` → tier 2
- `Rule` → tier 3
- `Directive` → tier 4
- `Notification` → tier 5
- NKP case (source_type = 'nkp_case', work_id = NULL) → tier 6

---

## Implementation guide

### 1. `app/retrieval/gated_orchestrator.py` — changes

#### 1a. Imports to add

```python
import re
```

Add to imports section. Also add this re-export (following the existing pattern for monkeypatch compatibility):

```python
from app.retrieval.eligibility_gate import eligible_chunk_ids as eligible_chunk_ids
```

Place after the existing two re-exports (`retrieve_postgres`, `validate_and_render`).

#### 1b. Remove `_REAL_MONOTONIC`

Delete the line:
```python
_REAL_MONOTONIC = time.monotonic
```
(line 19, was needed for `_graph_clock` which was removed in AGENT-2)

#### 1c. Module-level constants (add after `EXTRACTIVE_CHARS`)

```python
_WORK_TYPE_TIER: dict[str, int] = {
    "constitution": 1,
    "act": 2,
    "rule": 3,
    "regulation": 3,
    "directive": 4,
    "byelaw": 4,
    "notification": 5,
    "order": 5,
}

_DEVA_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

_CROSS_REF_RE = re.compile(
    r"(?:दफा|उपदफा)\s+([०-९\d]+(?:\([०-९\d]+\))?)"
    r"|अनुसूची\s+([०-९\d]+)"
)
```

#### 1d. `_authority_rank_hits`

```python
def _authority_rank_hits(
    hits: list[dict[str, Any]], conn: connection
) -> list[dict[str, Any]]:
    """Sort hits by (work_type tier ASC, rrf_score DESC). Attach tier and conflict_flag.

    Failure mode: any exception → return hits unchanged.
    """
    if not hits:
        return hits
    try:
        ids = [h["component_uri"] for h in hits]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id::text, w.work_type, c.source_type
                FROM chunks c
                LEFT JOIN work w ON w.id = c.work_id
                WHERE c.id::text = ANY(%(ids)s)
                """,
                {"ids": ids},
            )
            meta: dict[str, tuple[str, str]] = {
                str(row[0]): (str(row[1] or "").lower(), str(row[2] or "").lower())
                for row in cur.fetchall()
            }

        enriched: list[dict[str, Any]] = []
        for h in hits:
            wt, st = meta.get(h["component_uri"], ("", ""))
            tier = _WORK_TYPE_TIER.get(wt) or (6 if st == "nkp_case" else 99)
            enriched.append({**h, "work_type": wt or st, "tier": tier})

        enriched.sort(key=lambda x: (x["tier"], -x.get("score", 0.0)))

        seen_sections: dict[str, int] = {}
        for h in enriched:
            sec = h.get("section_number", "")
            if not sec:
                continue
            existing_tier = seen_sections.get(sec)
            if existing_tier is not None and existing_tier < h["tier"]:
                h["conflict_flag"] = True
            else:
                seen_sections[sec] = h["tier"]

        return enriched
    except Exception:
        return hits
```

#### 1e. `_resolve_cross_refs`

```python
def _resolve_cross_refs(
    hits: list[dict[str, Any]],
    as_of: date,
    conn: connection,
    max_additional: int = 5,
) -> list[dict[str, Any]]:
    """Find section cross-references in top-10 hits and fetch eligible co-chunks.

    Failure mode: any exception → return [].
    """
    if not hits:
        return []
    try:
        existing_ids = {h["component_uri"] for h in hits}
        eligible = list(eligible_chunk_ids(conn, as_of))
        if not eligible:
            return []

        additional: list[dict[str, Any]] = []

        for hit in hits[:10]:
            if len(additional) >= max_additional:
                break
            text = hit.get("text_ne", "")
            source_id = hit.get("document_source_id", "")
            if not source_id or not text:
                continue

            for m in _CROSS_REF_RE.finditer(text):
                if len(additional) >= max_additional:
                    break
                raw_num = (m.group(1) or m.group(2) or "").translate(
                    _DEVA_DIGIT_MAP
                ).strip()
                if not raw_num:
                    continue
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.id::text, c.chunk_text, c.span_sha256,
                               c.act_name, c.case_id, c.chunk_type,
                               c.section_number, d.source_id
                        FROM chunks c
                        JOIN documents d ON d.id = c.document_id
                        WHERE d.source_id = %(source_id)s
                          AND c.section_number = %(section_num)s
                          AND c.id::text = ANY(%(eligible)s)
                        LIMIT 1
                        """,
                        {
                            "source_id": source_id,
                            "section_num": raw_num,
                            "eligible": eligible,
                        },
                    )
                    row = cur.fetchone()
                if row and str(row[0]) not in existing_ids:
                    chunk_id = str(row[0])
                    existing_ids.add(chunk_id)
                    additional.append(
                        {
                            "component_uri": chunk_id,
                            "text_ne": row[1],
                            "text_hash": row[2],
                            "score": 0.0,
                            "work_title_ne": row[3] or row[4] or "",
                            "chunk_type": row[5],
                            "section_number": row[6] or "",
                            "document_source_id": str(row[7]) if row[7] else "",
                            "co_retrieved": True,
                        }
                    )

        return additional
    except Exception:
        return []
```

---

### 2. `app/retrieval/query_graph.py` — add two nodes + update graph

Add these two node functions before `build_graph()`:

```python
def authority_ranker_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    ranked = _orch._authority_rank_hits(state["all_hits"], conn)
    return {"all_hits": ranked}


def cross_ref_resolver_node(state: QueryState, config: RunnableConfig) -> dict[str, Any]:
    conn = config["configurable"]["conn"]
    additional = _orch._resolve_cross_refs(
        state["all_hits"], state["session_as_of"], conn
    )
    if not additional:
        return {}
    return {"all_hits": state["all_hits"] + additional}
```

Update `build_graph()`:

```python
def build_graph() -> Any:
    builder: StateGraph = StateGraph(QueryState)
    builder.add_node("fact_extractor", fact_extractor_node)
    builder.add_node("retrieve_generate", retrieve_generate_node)
    builder.add_node("authority_ranker", authority_ranker_node)
    builder.add_node("cross_ref_resolver", cross_ref_resolver_node)
    builder.add_node("validate", validate_node)
    builder.add_node("assemble", assemble_node)

    builder.add_edge(START, "fact_extractor")
    builder.add_edge("fact_extractor", "retrieve_generate")
    builder.add_edge("retrieve_generate", "authority_ranker")
    builder.add_edge("authority_ranker", "cross_ref_resolver")
    builder.add_edge("cross_ref_resolver", "validate")
    builder.add_edge("validate", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile()
```

---

### 3. `tests/test_orchestrator.py` — 5 new tests

Add these at the bottom of the file:

```python
def test_authority_rank_hits_sorts_by_tier() -> None:
    """Hits reordered by tier ASC, rrf_score DESC; tier attached to each hit."""

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            pass

        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("chunk-act", "Act", "act"),
                ("chunk-const", "Constitution", "act"),
                ("chunk-reg", "Rule", "regulation"),
            ]

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {"component_uri": "chunk-act", "score": 0.8, "section_number": "45", "text_ne": ""},
        {"component_uri": "chunk-const", "score": 0.6, "section_number": "3", "text_ne": ""},
        {"component_uri": "chunk-reg", "score": 0.9, "section_number": "12", "text_ne": ""},
    ]

    result = orchestrator._authority_rank_hits(hits, _Conn())

    assert result[0]["component_uri"] == "chunk-const"
    assert result[0]["tier"] == 1
    assert result[1]["component_uri"] == "chunk-act"
    assert result[1]["tier"] == 2
    assert result[2]["component_uri"] == "chunk-reg"
    assert result[2]["tier"] == 3


def test_authority_rank_hits_conflict_flag() -> None:
    """Same section_number covered by lower-tier chunk gets conflict_flag=True."""

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            pass

        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("chunk-act", "Act", "act"),
                ("chunk-rule", "Rule", "regulation"),
            ]

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {"component_uri": "chunk-act", "score": 0.8, "section_number": "10", "text_ne": ""},
        {"component_uri": "chunk-rule", "score": 0.9, "section_number": "10", "text_ne": ""},
    ]

    result = orchestrator._authority_rank_hits(hits, _Conn())

    act_hit = next(h for h in result if h["component_uri"] == "chunk-act")
    rule_hit = next(h for h in result if h["component_uri"] == "chunk-rule")
    assert "conflict_flag" not in act_hit
    assert rule_hit.get("conflict_flag") is True


def test_authority_rank_hits_failure_returns_unchanged() -> None:
    """Exception from conn → original hits returned unchanged."""
    hits = [{"component_uri": "x", "score": 0.5, "section_number": "", "text_ne": ""}]
    result = orchestrator._authority_rank_hits(hits, object())
    assert result is hits


def test_resolve_cross_refs_finds_section_reference(monkeypatch: Any) -> None:
    """दफा reference in text → co-retrieved chunk added with co_retrieved=True."""
    from app.retrieval import eligibility_gate

    monkeypatch.setattr(
        eligibility_gate,
        "eligible_chunk_ids",
        lambda conn, as_of: {"chunk-100", "chunk-456"},
    )

    class _Cur:
        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any) -> None:
            self._p = params

        def fetchone(self) -> tuple[Any, ...] | None:
            if self._p.get("section_num") == "456":
                return (
                    "chunk-456",
                    "दफा ४५६ को पाठ",
                    "hash456",
                    "Act Name",
                    None,
                    "dafa",
                    "456",
                    "src-001",
                )
            return None

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

    hits = [
        {
            "component_uri": "chunk-100",
            "text_ne": "यो दफा ४५६ मा उल्लेख भएको छ",
            "score": 0.8,
            "section_number": "100",
            "document_source_id": "src-001",
        }
    ]

    result = orchestrator._resolve_cross_refs(hits, date(2024, 1, 1), _Conn())

    assert len(result) == 1
    assert result[0]["component_uri"] == "chunk-456"
    assert result[0]["co_retrieved"] is True
    assert result[0]["section_number"] == "456"


def test_resolve_cross_refs_failure_returns_empty() -> None:
    """Exception from conn → empty list returned."""
    hits = [
        {
            "component_uri": "chunk-1",
            "text_ne": "दफा ४५ को प्रावधान",
            "score": 0.5,
            "section_number": "1",
            "document_source_id": "src-1",
        }
    ]
    result = orchestrator._resolve_cross_refs(hits, date(2024, 1, 1), object())
    assert result == []
```

---

## Why existing 7 tests pass unchanged

Both new functions have `except Exception: return <safe_default>` at the top level:
- `_authority_rank_hits(hits, object())` → `object()` has no `.cursor()` → `AttributeError` caught → returns `hits` unchanged
- `_resolve_cross_refs(hits, as_of, object())` → `eligible_chunk_ids(object(), as_of)` tries `object().cursor()` → `AttributeError` caught → returns `[]`

The new nodes call these functions and return `{}` or `{"all_hits": unchanged}` respectively — LangGraph state update with unchanged values passes through. ✓

---

## PS compliance

- **PS-6 (eligibility gate on every path)**: Cross-ref resolver calls `eligible_chunk_ids` before returning any co-chunk. Co-retrieved chunks are already filtered to eligible set. ✓
- **PS-16 (provisos co-retrieval)**: Cross-ref resolver handles explicit `दफा X` references. Implicit proviso co-retrieval (via `co_retrieve_parent_id` FK) is already handled by the ingestion chunker and not duplicated here.

---

## Zero-tolerance gates

- `repealed-as-current = 0` — unchanged. Authority ranker sorts only; no temporal eligibility change.
- `not-yet-effective-as-current = 0` — unchanged. Cross-ref resolver uses `eligible_chunk_ids` which applies the `effective_date_ad` check.

---

## Required checks

```bash
make test    # must show 62 passed (57 + 5 new)
make lint
```

---

## Commit authorship

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No AI attribution. No `Co-Authored-By` trailers.

---

## Return to Claude

Commit hash, changed files, `make test` output, `make lint` output, assumptions.
