# Task: PG-A — RAGAS eval rework (fix LangChain version conflict)

## Branch
`PG-A/ragas-eval` (already checked out)

## What Kimi already delivered — keep everything EXCEPT ragas_eval.py
All slice files, golden sets, metrics, and Makefile changes are correct and must not be touched.
The only file that needs to be fixed is `app/eval/ragas_eval.py`.

## Root cause of the conflict
`LangchainLLMWrapper` (and `LangchainEmbeddingsWrapper`) are ragas's 0.2.x
LangChain integration, built for `langchain-openai 0.1.x`. Our pipeline pins
`langchain-openai==1.4.1` (a 1.x major version). These cannot coexist.

Additionally, ragas 0.2.x may import `langchain_community.chat_models.vertexai`
which was removed in `langchain-community==0.4.2`.

## Diagnosis step (run this first)
```bash
python3 -c "import ragas; print('ragas ok')"
python3 -c "from ragas.metrics import Faithfulness; print('metrics ok')"
```

**If `import ragas` fails** with the vertexai error:
→ Add `langchain-google-vertexai>=2.0.0` to `requirements.txt` (it satisfies the
  missing `langchain_community.chat_models.vertexai` import chain in langchain-community 0.4.x).
  Re-run the test until `import ragas` succeeds.

**If `import ragas` succeeds but `from ragas.llms import LangchainLLMWrapper` fails:**
→ The fix below is sufficient — do NOT add langchain-google-vertexai.

## The fix — rewrite `app/eval/ragas_eval.py`

Replace the entire file with this implementation. It uses `openai.AsyncOpenAI`
directly — no `LangchainLLMWrapper`, no `LangchainEmbeddingsWrapper`, no
langchain-openai dependency in the eval path.

```python
from __future__ import annotations

import asyncio
from typing import Any, Sequence

from langchain_core.outputs import Generation, LLMResult
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM


class _OpenAIEvalLLM(BaseRagasLLM):
    """Direct openai.AsyncOpenAI wrapper — avoids langchain-openai version conflict."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI()
        self._model = model

    async def agenerate(
        self,
        prompts: Sequence[Any],
        n: int = 1,
        temperature: float = 1e-8,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        all_gens: list[list[Generation]] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                n=n,
                temperature=temperature,
                stop=stop or None,
            )
            all_gens.append(
                [Generation(text=c.message.content or "") for c in resp.choices]
            )
        return LLMResult(generations=all_gens)

    def generate_text(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 1e-8,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return asyncio.run(self.agenerate([prompt], n, temperature, stop, callbacks))

    @property
    def llm(self) -> "_OpenAIEvalLLM":
        return self


class _OpenAIEvalEmbeddings(BaseRagasEmbeddings):
    """Direct openai embeddings — avoids langchain-openai version conflict."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._client = AsyncOpenAI()
        self._model = model

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    async def aembed_query(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))


_llm = _OpenAIEvalLLM()
_embeddings = _OpenAIEvalEmbeddings()


def run_ragas(
    dataset: EvaluationDataset, metrics: list[Any], label: str
) -> dict[str, Any]:
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_llm,
        embeddings=_embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    scores: dict[str, Any] = result.to_pandas().mean(numeric_only=True).to_dict()
    print(f"\n=== {label} ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    return scores
```

## Notes on the implementation

- `langchain_core.outputs.LLMResult` and `Generation` ARE present in our env
  (`langchain==1.3.14` brings `langchain-core~=0.3`). These are stable base types.
- `openai` package is already available (it's a dependency of `langchain-openai`).
- `BaseRagasLLM` and `BaseRagasEmbeddings` are ragas's own abstract bases —
  no langchain-openai required.
- The `_llm` and `_embeddings` are module-level singletons; clients are
  constructed lazily by the AsyncOpenAI constructor (picks up `OPENAI_API_KEY`).

## If langchain_core.outputs imports fail

If `langchain_core.outputs.LLMResult` or `Generation` don't exist in the installed
langchain-core version, use this fallback — define them locally:

```python
from dataclasses import dataclass, field

@dataclass
class Generation:
    text: str

@dataclass
class LLMResult:
    generations: list[list[Generation]] = field(default_factory=list)
```

Only do this if the import fails. Prefer the real import.

## Allowed files to change
- `app/eval/ragas_eval.py` — rewrite as above
- `requirements.txt` — add `langchain-google-vertexai>=2.0.0` ONLY if `import ragas` fails without it
- No other files

## Required checks
```
python3 -c "import ragas; from ragas.metrics import Faithfulness; print('ok')"
make test
make lint
make eval-gates
make eval  # each slice should print "skipped" gracefully (no live DB in this env)
```

## Commit authorship (mandatory)
```
git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"
```
No AI attribution in any form.

## Return to Claude
1. Commit hash
2. Whether `langchain-google-vertexai` was needed (and why)
3. Whether `langchain_core.outputs` import worked or the fallback was used
4. `make test`, `make lint`, `make eval-gates` results
5. Any remaining risks
