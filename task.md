# Task PB-A — Phase B: Orchestrator + Degraded-mode Ladder

**Engineer:** Pi  
**Branch:** `phase-b/orchestrator-resilience`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Two deliverables wired end-to-end:

1. **Degraded-mode ladder** in the `/ask` path — three specific failure modes handled correctly without ever bypassing the authority store.
2. **Simple gated orchestrator** (`app/retrieval/gated_orchestrator.py`) — a router that classifies queries as simple or complex, runs multi-hop sub-queries each with their own per-claim as-of, enforces caps, and validates every claim through the gate before it ships.

The Core Invariants in `system-design.md §2` must hold on every path — including every degraded-mode fallback. No citation renders without a validation gate pass against Postgres.

---

## Acceptance criteria

1. **Postgres down** (`psycopg2.OperationalError` on `connect()`) → `/ask` returns HTTP 503 with body `{"message": "Authority store unavailable. No validated answers can be provided.", "retry_after": 60}`. No attempt to answer from the index alone.
2. **OpenSearch down** (`opensearchpy.ConnectionError` on `dumb_retriever.retrieve()`) → automatic fallback to `postgres_retriever.retrieve_postgres()`. Response is still validated through the gate. Abstains loudly if Postgres fallback also returns nothing.
3. **Model down** (any exception from `ChatOpenAI.invoke`) → extractive fallback: top eligible retrieval hit becomes a single claim (`claim = first 300 chars of text_ne`, `evidence_id = component_uri`). That claim is still passed through `validate_and_render()`. Response carries `"query_type": "extractive"` in the body.
4. **Simple query** → routed through existing single-hop path (same behavior as Phase A). `"query_type": "simple"` in body.
5. **Complex/comparative query** → orchestrator decomposes into max 3 sub-queries, each with its own `as_of` (defaults to session `as_of` if not detected). Each sub-query runs full pipeline: retrieve → model → validate. All validated claims fused into one response. `"query_type": "complex"` in body.
6. **Per-claim as-of in response:** Every item in `results` carries `"as_of"` (the as-of used to validate that specific claim), not just the session-level as-of.
7. **Orchestrator caps:** max 3 sub-queries, max 20 s wall-clock (abort remaining sub-queries and return what's been validated so far if exceeded).
8. Session as-of is the default on every hop; sub-queries may override it only if the classifier explicitly detects a comparative date range in the question.
9. `make test` green.
10. `make lint` green (include new files).
11. `make eval-gates` green (zero-tolerance gates untouched).

---

## New file: `app/retrieval/postgres_retriever.py`

Postgres ILIKE fallback — used when OpenSearch is unavailable.

```python
from __future__ import annotations
from datetime import date
from typing import Any
from psycopg2.extensions import connection
from app.retrieval.eligibility_gate import is_eligible

def retrieve_postgres(conn: connection, query: str, as_of: date, k: int = 5) -> list[dict[str, Any]]:
    """
    Fallback retriever: ILIKE search on expression.text_ne, then eligibility filter.
    Used only when OpenSearch is unavailable.
    """
    # Split query into tokens, build WHERE clause with AND ILIKE for each token (max 5 tokens).
    # Join expression → component → work to get component_uri and work titles.
    # Apply is_eligible(conn, component_uri, as_of) on each candidate.
    # Return same shape as dumb_retriever.retrieve():
    #   [{component_uri, text_ne, text_hash, score, work_title_ne}]
    # score = 1.0 for all hits (no ranking — this is a dumb fallback).
    # Limit the DB scan to 50 rows before eligibility filter.
```

Important: this retriever accepts `conn` as its first argument (caller already has the connection — reuse it; don't open a second).

---

## New file: `app/retrieval/gated_orchestrator.py`

```python
from __future__ import annotations
import time
from datetime import date
from typing import Any
from psycopg2.extensions import connection

from app.retrieval.dumb_retriever import retrieve as os_retrieve
from app.retrieval.postgres_retriever import retrieve_postgres
from app.retrieval.validation_gate import validate_and_render
from app.retrieval.eligibility_gate import is_eligible

WALL_CLOCK_CAP = 20.0  # seconds
MAX_SUBQUERIES = 3
EXTRACTIVE_CHARS = 300


def _try_retrieve(conn: connection, query: str, as_of: date, k: int = 5) -> list[dict[str, Any]]:
    """OpenSearch first; Postgres ILIKE fallback on ConnectionError."""
    try:
        return os_retrieve(query, as_of, k)
    except Exception as os_exc:
        # Only fall back on connection-class errors, not logic errors.
        import opensearchpy
        if isinstance(os_exc, (opensearchpy.ConnectionError, opensearchpy.TransportError)):
            return retrieve_postgres(conn, query, as_of, k)
        raise


def _model_claims(question: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Call model; return None on any failure (caller handles extractive fallback)."""
    import json, os
    from langchain_openai import ChatOpenAI
    context = "\n\n---\n\n".join(
        f"[{h['component_uri']}]\n{h['text_ne']}" for h in hits
    )
    system = (
        "You are Wakil-G. Answer using ONLY the provided context. "
        "Output JSON only:\n"
        '{"claims": [{"claim": "<answer text>", "evidence_id": "<component_uri>"}]}\n'
        'If context is insufficient: {"claims": [], "abstain": true}\n'
        "Do not write citations."
    )
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.0,
        )
        resp = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ])
        return json.loads(resp.content.strip())
    except Exception:
        return None


def _extractive_claim(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Top hit → single claim, no model."""
    if not hits:
        return []
    h = hits[0]
    return [{"claim": h["text_ne"][:EXTRACTIVE_CHARS], "evidence_id": h["component_uri"]}]


def _classify_and_decompose(question: str, session_as_of: date) -> list[dict[str, Any]]:
    """
    Returns a list of sub-queries:
      [{"subquery": str, "as_of": date}]

    Uses LLM to classify simple vs complex. Simple → one entry (original question,
    session_as_of). Complex → up to MAX_SUBQUERIES entries, each with its own as_of
    (detected date range, else session_as_of).

    Output JSON prompt:
      {"type": "simple"} or
      {"type": "complex", "subqueries": [{"q": "...", "as_of": "YYYY-MM-DD or null"}]}

    On any parse failure → treat as simple.
    """
    import json, os
    from langchain_openai import ChatOpenAI
    system = (
        "Classify the legal question as simple (single issue, single time point) or "
        "complex (multiple issues or comparative across time). "
        "Output JSON only:\n"
        '{"type": "simple"} OR\n'
        '{"type": "complex", "subqueries": [{"q": "<sub-question>", "as_of": "<YYYY-MM-DD or null>"}]}\n'
        f"Default as_of if not detected: {session_as_of.isoformat()}. Max {MAX_SUBQUERIES} sub-queries."
    )
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.0,
        )
        resp = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])
        parsed = json.loads(resp.content.strip())
        if parsed.get("type") == "complex":
            subs = parsed.get("subqueries", [])[:MAX_SUBQUERIES]
            result = []
            for s in subs:
                try:
                    sq_as_of = date.fromisoformat(s["as_of"]) if s.get("as_of") else session_as_of
                except ValueError:
                    sq_as_of = session_as_of
                result.append({"subquery": s["q"], "as_of": sq_as_of})
            return result if result else [{"subquery": question, "as_of": session_as_of}]
    except Exception:
        pass
    return [{"subquery": question, "as_of": session_as_of}]


def answer(
    question: str,
    session_as_of: date,
    conn: connection,
) -> dict[str, Any]:
    """
    Main orchestrator entry point. Called from /ask after auth.
    Never raises — returns a dict with abstained=True on unrecoverable errors.
    Postgres-down exceptions bubble up to /ask (caller converts to 503).
    """
    start = time.monotonic()

    subqueries = _classify_and_decompose(question, session_as_of)
    query_type = "simple" if len(subqueries) == 1 else "complex"

    all_results: list[dict[str, Any]] = []

    for sq in subqueries:
        if time.monotonic() - start > WALL_CLOCK_CAP:
            break

        sq_q = sq["subquery"]
        sq_as_of = sq["as_of"]

        hits = _try_retrieve(conn, sq_q, sq_as_of)
        if not hits:
            continue

        parsed = _model_claims(sq_q, hits)

        if parsed is None:
            # Model down → extractive fallback
            claims = _extractive_claim(hits)
            query_type = "extractive"
        elif parsed.get("abstain") or not parsed.get("claims"):
            continue
        else:
            claims = parsed["claims"]

        validated = validate_and_render(claims, sq_as_of, conn)
        # Tag each result with its declared as_of
        for r in validated:
            r["as_of"] = sq_as_of.isoformat()
        all_results.extend(validated)

    abstained = len(all_results) == 0
    return {
        "as_of": session_as_of.isoformat(),
        "query_type": query_type,
        "abstained": abstained,
        "results": all_results,
    }
```

---

## Modify: `app/main.py`

Replace the current `/ask` handler body with a call to `orchestrator.answer()` wrapped in degraded-mode handling.

```python
from app.retrieval.gated_orchestrator import answer as orchestrator_answer
import psycopg2

@app.post("/ask")
async def ask_question(req: AskRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    supabase_helper = SupabaseHelper()
    user_id = supabase_helper.get_user_id(authorization)
    if supabase_helper.check_daily_quota(user_id):
        raise HTTPException(status_code=404, detail={"message": "Daily quota reached."})

    as_of = req.as_of or date.today()

    try:
        with connect() as conn:
            return orchestrator_answer(req.question, as_of, conn)
    except psycopg2.OperationalError:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Authority store unavailable. No validated answers can be provided.",
                "retry_after": 60,
            },
        )
```

Remove the old `_claims_from_model` helper and the old imports (`retrieve`, `validate_and_render`) from main — those now live in the orchestrator. Keep `connect`, `SupabaseHelper`, `logger`, `load_dotenv`, FastAPI setup, CORS, `GET /`.

---

## New test file: `tests/test_degraded_modes.py`

Mock-based tests — no live DB or OpenSearch. Cover:

1. **Postgres down** — `connect()` raises `psycopg2.OperationalError` → HTTP 503.
2. **OpenSearch down, Postgres fallback returns hits** — `os_retrieve` raises `opensearchpy.ConnectionError`, `retrieve_postgres` returns a hit, model returns a claim, validation gate passes → 200 with a result.
3. **OpenSearch down, Postgres fallback returns nothing** → `abstained: True`.
4. **Model down (extractive)** — `_model_claims` returns `None` → top hit becomes extractive claim, still validated through gate → `query_type: "extractive"`.

---

## New test file: `tests/test_orchestrator.py`

Mock-based. Cover:

1. **Simple query** — classifier returns simple → single sub-query, session as_of used → `query_type: "simple"`.
2. **Complex query** — classifier returns 2 sub-queries with different as_ofs → 2 retrieve+validate cycles, per-result as_of in response.
3. **Wall-clock cap** — mock `time.monotonic` to exceed cap after first sub-query → only first sub-query's results returned.
4. **Classifier failure** — LLM call raises exception → falls back to simple (original question, session_as_of).

---

## Makefile — update lint paths

Add to all three `python3 -m ruff` / `python3 -m mypy` lines:
- `app/retrieval/postgres_retriever.py`
- `app/retrieval/gated_orchestrator.py`
- `tests/test_degraded_modes.py`
- `tests/test_orchestrator.py`

---

## Do NOT touch

- `app/retrieval/dumb_retriever.py` — no changes.
- `app/retrieval/eligibility_gate.py` — no changes.
- `app/retrieval/validation_gate.py` — no changes.
- `app/retrieval/advanced_retriever.py`, `query_processor.py`, `retrieval_orchestrator.py` — untouched.
- `app/eval/gates.py` — must stay green.
- `migrations/` — no changes.

---

## Forbidden

- No path may return a citation without passing through `validate_and_render()` against Postgres.
- Postgres-down must never silently degrade to index-only answers — it must 503.
- Model self-abstention is advisory only; server gate owns abstention.
- No new pip dependencies.
- No Pinecone, Cohere, or LangGraph.
- Orchestrator tools must be read-only; no mutation tools may exist in the orchestrator's scope.

---

## PS requirements in scope

| PS | Enforcement |
|---|---|
| PS-6 | Per-claim as_of: each result carries its declared as_of; each validate call uses that as_of |
| PS-7 | Model abstention advisory; `validate_and_render()` owns abstention on every path |
| PS-11 | Postgres-down → HTTP 503, no fallback to index-only answers |

## Zero-tolerance gates (must stay 0)
- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

---

## Required checks

```
make lint
make test
make eval-gates
```

---

## Commit authorship

Every commit MUST use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No `Co-Authored-By: Claude`, no "Generated with Claude", no AI attribution of any kind.

---

## Return to Claude (via Prakash)

- Commit hash
- Changed/created files list
- `make lint` output
- `make test` output
- `make eval-gates` output
- Assumptions and remaining risks
