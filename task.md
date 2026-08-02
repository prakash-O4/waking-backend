# Task PA-A — Phase A: Wire `/ask` with Bitemporal Gated Pipeline

**Engineer:** Pi  
**Branch:** `phase-a/statute-path`  
**Base branch:** `dev`  
**Status:** ASSIGNED

---

## Objective

Replace the old Pinecone/Cohere retrieval path in `app/main.py /ask` with the bitemporal gated pipeline built in Phase 0. The endpoint must enforce the eligibility gate (pre-retrieval) and validation gate (server-side) on every query, accept an explicit `as_of` date per request, and return claims validated against the authority store — never model-written citations.

---

## Acceptance criteria

1. `POST /ask` accepts `{"question": "...", "as_of": "YYYY-MM-DD"}` (`as_of` optional; defaults to today).
2. Retrieval runs through `dumb_retriever.retrieve(query, as_of, k=5)` — eligibility gate enforced pre-retrieval.
3. Model emits `{"claims": [{"claim": "...", "evidence_id": "<component_uri>"}]}` JSON only. It does not write citations.
4. All claims pass through `validate_and_render(claims, as_of, conn)` — validation gate enforced server-side.
5. Response is a JSON object:
   ```json
   {
     "as_of": "YYYY-MM-DD",
     "abstained": false,
     "results": [
       {
         "claim": "...",
         "abstained": false,
         "citation": {
           "component_uri": "...",
           "work_title_ne": "...",
           "work_title_en": "...",
           "as_of": "YYYY-MM-DD",
           "source_kind": "...",
           "ocr_confidence": null
         }
       }
     ]
   }
   ```
6. If no eligible hits from retrieval: `{"as_of": "...", "abstained": true, "results": []}`.
7. Auth check (Supabase JWT + daily quota) is preserved.
8. `make test` green.
9. `make lint` green (including `app/main.py`).
10. `make eval-gates` green (zero-tolerance gates untouched).

---

## What to change in `app/main.py`

### Remove
- The `initialize_pinecone()` function and its module-level call (or any at-startup Pinecone init).
- The `create_advanced_retriever()` function.
- The `from app.retrieval import RetrievalOrchestrator` import inside `/ask`.
- The `PineconeVectorStore` / LangChain Pinecone imports.
- The streaming SSE response (drop SSE for Phase A — return plain JSON).
- The `qa_prompt` and `qa_system_prompt` (replace with the claims-only prompt below).
- The `TokenBuffer` usage (no longer needed without SSE streaming).

### Keep
- CORS middleware, `GET /` health check.
- `SupabaseHelper` auth + daily quota check.
- `load_dotenv()`, FastAPI app setup.

### Add / replace

**Request model:**
```python
from datetime import date

class AskRequest(BaseModel):
    question: str
    as_of: date | None = None
```

**Imports to add:**
```python
from app.retrieval.dumb_retriever import retrieve
from app.retrieval.validation_gate import validate_and_render
from app.authority.writer import connect
```

**New `/ask` handler:**
```python
@app.post("/ask")
async def ask_question(req: AskRequest, authorization: str = Header(None)):
    supabase_helper = SupabaseHelper()
    user_id = supabase_helper.get_user_id(authorization)
    if supabase_helper.check_daily_quota(user_id):
        raise HTTPException(status_code=404, detail={"message": "Daily quota reached."})

    as_of: date = req.as_of or date.today()

    hits = retrieve(req.question, as_of, k=5)
    if not hits:
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    context_parts = [f"[{h['component_uri']}]\n{h['text_ne']}" for h in hits]
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.0)
    system = (
        "You are Wakil-G. Answer using ONLY the provided context. "
        "Output JSON only — no markdown, no explanation outside the JSON:\n"
        '{"claims": [{"claim": "<answer text>", "evidence_id": "<component_uri from context>"}]}\n'
        "If context is insufficient: {\"claims\": [], \"abstain\": true}\n"
        "Do not write citations. Do not include anything not in the context."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {req.question}"

    try:
        response = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        raw = response.content.strip()
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"Model call or parse failed: {e}")
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    if parsed.get("abstain") or not parsed.get("claims"):
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    with connect() as conn:
        validated = validate_and_render(parsed["claims"], as_of, conn)

    return {"as_of": as_of.isoformat(), "abstained": False, "results": validated}
```

---

## Makefile — update lint paths

Add `app/main.py` to the `lint` target's file list (the three `python3 -m ruff` / `python3 -m mypy` lines).

---

## Test file — `tests/test_ask_pipeline.py`

Write unit tests that mock `retrieve`, `validate_and_render`, `ChatOpenAI`, and `SupabaseHelper`. Cover:
1. Happy path — model returns valid claims, validation gate passes → response has `abstained: False` + citation.
2. No eligible hits → `abstained: True`.
3. Model returns `{"abstain": true}` → `abstained: True`.
4. Validation gate abstains a claim (hash mismatch / not eligible) → that claim has `abstained: True`.

Do NOT require a live DB or OpenSearch in tests.

---

## Do NOT touch

- `app/retrieval/advanced_retriever.py`
- `app/retrieval/query_processor.py`
- `app/retrieval/retrieval_orchestrator.py`
- `app/ingestion/` — any existing file
- `migrations/001_bitemporal_schema.sql`
- `app/eval/gates.py` — must stay green

---

## Forbidden

- No Pinecone imports in any new or modified code path.
- No Cohere imports in any new or modified code path.
- Model must emit `claims + evidence_ids` only — no citations from the model.
- No path skips the eligibility gate.
- No path bypasses the validation gate.
- No new pip dependencies.

---

## PS requirements in scope

| PS | Enforcement |
|---|---|
| PS-6 | `as_of` is per-request, passed to every retrieve + validate call |
| PS-7 | Model `"abstain": true` is advisory; server `validated` list owns abstention |
| PS-3 | Citations rendered only from `source_publication + work` metadata in `validate_and_render()` |

## Zero-tolerance gates

- `repealed-as-current = 0`
- `not-yet-effective-as-current = 0`

Run `make eval-gates` and include output.

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
- Sample: `curl -X POST http://localhost:8000/ask -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"question":"What are the penalties under the Prevention of Corruption Act?","as_of":"2024-01-01"}'` output
- Assumptions and remaining risks
