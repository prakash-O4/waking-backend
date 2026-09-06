# task.md

## AGENT-41 — fix silently-failing Gemini calls (deprecated model, zero logging)

### Context / root cause (already diagnosed — do not re-derive, verify and fix)

`gemini-2.5-flash` is hardcoded in three places and is now rejected by Google:

```
GoogleModelNotFoundError: Error calling model 'gemini-2.5-flash' (NOT_FOUND): 404 NOT_FOUND.
{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer
available to new users. ...'}}
```

Reproduced directly against the live API key on 2026-09-06. All three call sites wrap the
call in a bare `except Exception: return <fallback>` with **no logging at all** —
`app/retrieval/gated_orchestrator.py` and `app/retrieval/postgres_retriever.py` import no
logger — so this has been failing silently in production since the model was deprecated:

1. `app/retrieval/postgres_retriever.py:86` `translate_query()` — silently returns `None`.
   Effect: the AGENT-39 dual-query normalization (Nepali translation of English/Roman
   queries) **has never actually run**. Every non-Devanagari query goes to vector search
   untranslated, against an embedding space built on Nepali text — this is why retrieval
   for "What does the labor law says?" pulled in an unrelated Cooperatives Act chunk
   (top vector score was only 0.405).
2. `app/retrieval/gated_orchestrator.py:104` `_structured_claims()` — NOT affected, this one
   uses Azure (`AZURE_OPENAI_LLM_DEPLOYMENT`), not Gemini. Do not touch.
3. `app/retrieval/gated_orchestrator.py:185` `_compose_answer()` (line ~244, model param) —
   silently returns `None`. Effect: the pipeline falls back to the raw extractive claims
   shape instead of the composed answer (`degraded_mode: ["reasoner_fallback:extractive"]`
   on every real query right now).
4. `app/retrieval/gated_orchestrator.py:293` `_fact_extract()` (line ~330, model param) —
   silently returns the passthrough fallback (raw English query, no facts/issue_queries).

Because the exception happens inside `llm.invoke()`, before `_lf_gen_end(gen, ...)` runs,
the Langfuse trace shows the generation's `input` but never its `output` — that's a
symptom of this bug, not a separate Langfuse instrumentation problem. Do not try to "fix"
Langfuse; fixing the underlying call fixes the trace.

### This is not a one-line rename — verify empirically before committing to a model

I tested replacement candidates directly against the real API. Findings, so you don't have
to rediscover them, but **do re-verify yourself against the real API before shipping** —
this is the same class of mistake as AGENT-40 (an assumption about third-party runtime
shape that turned out wrong):

- `gemini-2.5-flash` — 404, dead.
- `gemini-2.5-flash-lite` — 404, dead.
- `gemini-3.6-flash` (Google's own suggested replacement in the 404 message) — works, but:
  - `resp.content` came back as a **list of content blocks**
    (`[{'type': 'text', 'text': '...', 'extras': {...}}]`), not a plain string, on one call.
  - On a different call (shorter `max_output_tokens`, no system message) it came back
    **empty** (`resp.content == []`, `resp.text == ''`) — looks like a reasoning/thinking
    token budget eating the output budget before any visible text is produced. This means
    a naive swap can silently trade "hard failure" for "silent empty success," which is
    worse. You must confirm real prompts (the actual fact-extraction / compose-answer
    system prompts, not a toy "reply with ok") reliably produce non-empty, parseable text
    at the `max_output_tokens` values already set in the code (1000 / unset / 300), and
    bump them if needed.
  - It also ignores `temperature` ("uses fixed sampling defaults") — flag this explicitly
    in your completion report; determinism of legal-claim generation may be affected and
    that's a judgment call for Claude to review, not yours to silently accept.
  - `gemini-flash-latest` — returned empty content in a quick test; treat as unverified,
    don't assume it's better just because the name says "latest."

Pick whichever currently-available flash-tier model you empirically confirm produces
correct, non-empty, parseable output for all three real prompts in this codebase. Prefer
one that returns plain-string `.content`/`.text` if you find one that does — simpler and
avoids the parsing change below. If none do, implement the parsing change below regardless
of which model you pick (defensive either way).

### Required changes

1. **Config**: add `GEMINI_MODEL: str = "<your verified model>"` to `Settings` in
   `app/config.py`, following the existing `AZURE_OPENAI_LLM_DEPLOYMENT` pattern (see line
   45). Replace all three hardcoded `model="gemini-2.5-flash"` strings (and the
   `_lf_gen_start(..., "gemini-2.5-flash", ...)` name arg at gated_orchestrator.py:252/339)
   with `s.GEMINI_MODEL` / the settings value. One source of truth, not three literals.

2. **Robust text extraction**: add one small helper (e.g. `_llm_text(resp) -> str` — pick
   one file, e.g. `app/utils/loggers.py` or a new tiny `app/utils/llm.py`, whichever fits
   without creating a new top-level module for one function) that returns the plain text
   regardless of whether `resp.content` is a string or a list of content blocks. Use
   LangChain's `AIMessage.text` property if your verified langchain-core version supports
   it as a property (check — it may still be a deprecated method in this pin); fall back to
   extracting `block["text"]` from list-of-dict content otherwise. Use this helper at all
   three call sites in place of `str(resp.content)`.

3. **Real logging on failure** — this is the actual production-safety fix, independent of
   which model you pick. Add `from app.utils.loggers import logger` to both
   `gated_orchestrator.py` and `postgres_retriever.py` (already the established pattern —
   see `app/retrieval/advanced_retriever.py` for the convention:
   `logger.warning(f"... failed: {e}")` inside an except block). Change all three
   `except Exception:` blocks to `except Exception as e:` and log a `logger.warning(...)`
   (not `.exception()` — these are expected-fallback paths, not crashes) naming the
   function and the exception, before returning the fallback. Never let an LLM call fail
   silently again.

### Explicitly out of scope — do not touch

- The reranker `passthrough` fallback (`app/retrieval/reranker.py`) — separate, unrelated
  issue (flashrank import/model failure), not part of this task.
- `_structured_claims()` (Azure-based reasoner) — not broken, uses a different provider.
- Any change to the actual retrieval ranking/scoring logic — this task only restores the
  translation/fact-extraction/answer-composition calls to actually running; it does not
  change what happens once they do.

### Tests

- Extend/add unit tests for `translate_query`, `_fact_extract`, `_compose_answer`: mock a
  failing `llm.invoke` and assert (a) the existing fallback return value is unchanged, and
  (b) `logger.warning` is called (use `caplog` or monkeypatch the logger). Mock a
  successful call returning list-of-content-block shape and assert the new text-extraction
  helper parses it correctly — this is the shape that actually broke, so a test using only
  a plain-string mock would miss the real bug (same lesson as AGENT-40's `ClaimsResponse`
  mock mismatch — mock the real runtime shape, not the convenient one).

### Verification (required before reporting done)

- Run the three functions against the **real** Gemini API (not mocks) with realistic
  prompts and confirm non-empty, correctly-parsed output for each. Paste the actual
  responses in your completion report — "tests pass" is not sufficient here given how this
  bug hid for as long as it did.
- Run a real end-to-end query through `/ask` or `/ask/stream` with a non-Devanagari
  question (e.g. "What does the labor law says?" or a Romanized query) and confirm the
  `degraded_mode` field no longer contains `reasoner_fallback:extractive`, and that
  retrieval is now translated (check `eligibility_gate` / retrieval span metadata —
  `translation_ran` should be `true` for a non-Devanagari query).
- Targeted tests only (`.venv/bin/python -m pytest tests/test_helpers.py
  tests/<relevant new/existing test files> -v`, plus scoped ruff/mypy on changed files) —
  do not run the full `make test` suite unless asked.
