# PE-A/fix — Provider-agnostic LLM via LangChain 1.3.0

**Branch:** `pe-a/langchain-llm`
**Engineer:** Kimi
**Base:** `dev` (already merged PE-A implementation)

---

## Context

PE-A hardcoded `anthropic` SDK and `claude-haiku-4-5-20251001` in two files.
That was a Claude orchestrator mistake — Prakash's stack is provider-agnostic
(OpenAI, Gemini, Kimi, etc.). This task replaces the direct Anthropic SDK usage
with LangChain 1.3.0's `init_chat_model`, which is provider-agnostic and
configured entirely from a single env var.

No gate logic, no schema, no retrieval path is touched. This is a pure
LLM-client swap in two ingestion-side files.

---

## Files to change

### `requirements.txt`

- Change `langchain>=0.2.0` to `langchain>=1.3.0`
- Change `langchain-community==0.2.18` to `langchain-community>=1.3.0` (hard pin conflicts with langchain 1.x)
- Change `langchain-openai==0.1.25` to `langchain-openai>=0.3.0` (find latest compatible with langchain 1.3.0 and pin it)
- Remove `anthropic>=0.40.0` — added by PE-A solely for the two files being rewritten here; nothing else imports it
- Keep everything else untouched

Run `pip install "langchain>=1.3.0" langchain-openai langchain-community` to find compatible versions. Pin the resolved versions.

### `app/config.py`

Add one field to `Settings`:

```python
LLM_MODEL: str = "openai:gpt-4o-mini"
```

The working tree already has `extra = "ignore"` in `Settings.Config` — commit that fix too (it prevents Settings() from crashing on legacy .env keys). No other changes to config.py.

### `app/ingestion/metadata_enricher.py` — rewrite LLM transport only

Replace the `anthropic`-based `_get_client()` / `_call_haiku()` / `HAIKU_MODEL` with LangChain 1.3.0. Keep all existing logic (JSON parsing, prompt construction, `enrich_law_chunks`, `enrich_nkp_chunks`) exactly as-is — only the transport layer changes.

LangChain 1.3.0 API (lazy init, no module-level import):

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from app.config import get_settings

def _get_llm():
    settings = get_settings()
    return init_chat_model(settings.LLM_MODEL, temperature=0)

def _call_llm(prompt: str) -> str:
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    return response.content  # str
```

Keep the existing manual retry loop in `_call_llm` (max 3 retries, exponential backoff) — catch `Exception` broadly since the exception type varies by provider. Remove `HAIKU_MODEL`, `_get_client()`, and all `import anthropic` lines. `BATCH_API_MIN_DOCUMENTS` constant and its comment can stay (future work).

### `app/ingestion/pii_redactor.py` — replace `_haiku_second_pass` only

Replace the `_haiku_second_pass` method body with a LangChain call. Everything else (Stage 1, Stage 3, `_verify`, `RedactionVerificationError`, all public interface) stays byte-for-byte identical.

```python
def _haiku_second_pass(self, text: str, appellant: str, respondent: str) -> str:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    from app.config import get_settings

    llm = init_chat_model(get_settings().LLM_MODEL, temperature=0,
                          timeout=self._llm_timeout)
    prompt = (
        f"Given these party names: [{appellant}], [{respondent}] - identify any "
        "variant mentions (abbreviated names, honorifics, partial names) in the "
        "following text and return a JSON list of spans to replace. Format: "
        '[{"original": "...", "replacement": "[[vadI]]"}]. Use [[vadI]] for the '
        "appellant and [[prativadI]] for the respondent. Return ONLY the JSON "
        f"list.\n\nText:\n{text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content
    spans = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
    for span in spans:
        original = str(span.get("original") or "")
        replacement = str(span.get("replacement") or APPELLANT_PLACEHOLDER)
        if original:
            text = text.replace(original, replacement)
    return text
```

Keep the Devanagari placeholders ([[वादी]] / [[प्रतिवादी]]) exactly as they are in the existing file — the snippet above uses ASCII for display only; use the actual Devanagari strings from the existing code. Remove `HAIKU_MODEL` constant and all `import anthropic` lines from this file.

---

## Files NOT to touch

Everything in `app/retrieval/`, `app/authority/`, `app/main.py`, `app/eval/`, `migrations/`, `scripts/`. Tests require no changes (all LLM paths are guarded by `enable_llm=False` in the existing suite).

---

## Required checks

```bash
make lint    # ruff + mypy must be clean
make test    # all 35 must pass (2 skipped is OK)
```

---

## Commit authorship

Every commit must use:
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No Co-Authored-By, no AI attribution.

---

## Return to Claude when done

1. Commit hash
2. `make lint` and `make test` output
3. Exact pinned versions added to requirements.txt
4. Any assumption not in this brief
