# Task: RET-B — Dual-path cross-lingual query translation

**Branch:** `ret/query-translation`  
**Base:** `dev`  
**Engineer:** Pi  

---

## Objective

Users query in English, Romanized Nepali, or mixed language, but the corpus chunks are embedded in Devanagari Nepali. `text-embedding-3-large` has a measurable cross-lingual gap for low-resource Indic languages, so raw cross-lingual embedding similarity is weaker than it should be.

**Fix:** before embedding, translate non-Nepali queries to Devanagari Nepali using **Gemini 2.5 Flash** (lightweight, fast, published Nepali quality data, GEMINI_API_KEY already in `.env`). Then run retrieval on **both** the original query and the translated query and fuse the four result lists with RRF. This is the dual-path pattern — it hedges against translation errors on legal terminology while capturing the cross-lingual semantic boost.

**New dependency:** `langchain-google-genai` — extends the existing LangChain framework already in `requirements.txt`. One new pip package, one new config field (`GEMINI_API_KEY`).

---

## Acceptance criteria

1. Pure Devanagari queries bypass translation entirely (no extra LLM call).
2. English / Romanized / mixed queries are translated to Devanagari Nepali before the embed step.
3. If translation fails (API error, empty response, key unset) the function returns `None` and retrieval falls back to the single-path original query — no exception propagated.
4. `retrieve_postgres` runs vector and lexical search for **both** original and translated query (when translation is available), then fuses all ranked lists via the existing `_rrf()` function.
5. Existing Langfuse span metadata is extended to record `translation_ran: bool`.
6. `make test` green, `make lint` clean.
7. Romanized eval slice (`python -m app.eval.romanized_slice`) runs without error.

---

## Exact scope

**Only these files may change:**

- `requirements.txt` — add `langchain-google-genai>=2.0`
- `app/config.py` — add `GEMINI_API_KEY: str = ""`
- `app/retrieval/postgres_retriever.py` — add `_is_devanagari()`, `translate_query()`, extend `retrieve_postgres()` for dual-path
- `tests/test_retrieval.py` — add tests for new functions and dual-path

Do NOT modify `gated_orchestrator.py`, `eligibility_gate.py`, `validation_gate.py`, or any eval slice file. No new files.

---

## Implementation guide

### 1. `requirements.txt`

Add one line (group it near the other langchain-* packages):

```
langchain-google-genai>=2.0
```

### 2. `app/config.py` — Settings class

Add one field inside the `Settings` class, grouped with the other API keys:

```python
GEMINI_API_KEY: str = ""
```

### 3. `_is_devanagari(text: str) -> bool`

Add to `postgres_retriever.py`:

```python
def _is_devanagari(text: str) -> bool:
    count = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    return count / max(len(text), 1) > 0.5
```

Threshold 0.5: majority Devanagari → treat as pure Nepali → skip translation.

### 4. `translate_query(query: str) -> str | None`

Add to `postgres_retriever.py`:

```python
def translate_query(query: str) -> str | None:
    if _is_devanagari(query):
        return None
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=s.GEMINI_API_KEY,
            temperature=0.0,
            max_output_tokens=300,
        )
        resp = llm.invoke([
            {
                "role": "system",
                "content": (
                    "Translate the following legal query to formal Nepali in Devanagari script. "
                    "Output only the translated text. Do not add explanations."
                ),
            },
            {"role": "user", "content": query},
        ])
        translated = str(resp.content).strip()
        return translated if translated else None
    except Exception:
        return None
```

### 5. Extend `retrieve_postgres` — dual-path

After the existing `_preprocess` call and before `eligible_chunk_ids`:

```python
query_ne = translate_query(query)
```

After computing `qvec = _embed_query(query)`, add:

```python
qvec_ne = _embed_query(query_ne) if query_ne else None
```

Vector search: run the existing query once for `qvec`. If `qvec_ne` is not None, run a **second identical SQL query** substituting `qvec_ne` for `qvec` — call the results `vector_rows_ne`.

Lexical search: run the existing tsvector query once for `query`. If `query_ne` is not None, run a **second identical SQL query** substituting `query_ne` for `query` — call the results `lexical_rows_ne`.

Merge all into `rows` dict (dedup by chunk_id as before). Build RRF input:

```python
ranked_lists = [
    [str(r[0]) for r in vector_rows],
    [str(r[0]) for r in lexical_rows],
]
if qvec_ne is not None:
    ranked_lists.append([str(r[0]) for r in vector_rows_ne])
if query_ne is not None:
    ranked_lists.append([str(r[0]) for r in lexical_rows_ne])
rrf_scores = _rrf(ranked_lists)
```

The rest of the function (relevance gate, rerank, final fetch) is **unchanged**.

Add `translation_ran=query_ne is not None` to the existing `_span(trace, "eligibility_gate", ...)` metadata dict.

---

## System Design refs

- §8 Query plane — this change is in the "understanding" phase, before the eligibility gate
- §2 Core Invariant #2 — eligibility gate is untouched; dual-path retrieval still goes through the same gate
- PS-8 — Romanized Nepali is the primary eval slice this change serves

**No invariant is weakened.** The eligibility gate, validation gate, and model output contract are all unchanged.

---

## Zero-tolerance gates guarded

None of the zero-tolerance gates (`repealed-as-current`, `not-yet-effective-as-current`, `overruled-as-good-law`) are in scope. They must remain at 0 — this change cannot affect them (retrieval preprocessing doesn't touch temporal eligibility or citation rendering).

---

## Tests required (`tests/test_retrieval.py`)

Add to the existing file — do not create a new file.

1. `test_is_devanagari_pure_nepali` — Devanagari string → True
2. `test_is_devanagari_pure_english` — ASCII string → False
3. `test_is_devanagari_mixed_romanized` — romanized Nepali (ASCII) → False
4. `test_translate_query_skips_devanagari` — `translate_query("दफा १")` returns None without calling the LLM
5. `test_translate_query_returns_none_on_api_failure` — monkeypatch `ChatGoogleGenerativeAI` to raise, assert returns None
6. `test_translate_query_skips_when_key_unset` — monkeypatch settings with empty `GEMINI_API_KEY`, assert returns None without calling LLM
7. `test_dual_path_uses_four_lists_when_translated` — monkeypatch `r.translate_query` to return a Nepali string, `_embed_query` to return a fixed vector, capture the lists passed to `_rrf`, assert 4 lists
8. `test_dual_path_falls_back_to_single_when_translation_none` — monkeypatch `r.translate_query` to return None, assert `_rrf` receives 2 lists

Use the existing `patch_common` + `Conn`/`Cursor` pattern for tests 7 and 8.

---

## Required checks

```bash
make test
make lint
python -m app.eval.romanized_slice   # must run without error
```

---

## Commit authorship

Every commit must be authored as:

```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```

No AI attribution. No `Co-Authored-By` trailers. No "Generated with Claude" lines.

---

## Return to Claude

When done, return: commit hash, list of changed files, output of `make test` and `make lint`, and any assumptions made.
