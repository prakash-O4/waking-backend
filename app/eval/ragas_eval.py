from __future__ import annotations

import os
from typing import Any, Sequence, cast

from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI, OpenAI, RateLimitError
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM, ChatGeneration, LLMResult
from ragas.run_config import RunConfig

from app.config import get_settings


def _openai_model_name(settings_model: str) -> str:
    """Strip a provider prefix (e.g. 'openai:gpt-4o-mini') if present."""
    if ":" in settings_model:
        return settings_model.split(":", 1)[1]
    return settings_model


def _messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages to OpenAI chat-completion format."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.type
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "system":
            role = "system"
        content = m.content
        if isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _llm_result_from_openai(response: Any, n: int) -> LLMResult:
    """Build a ragas LLMResult from an OpenAI chat completion response."""
    choice = response.choices[0]
    message = choice.message
    generations: list[list[ChatGeneration]] = [
        [
            ChatGeneration(
                message=message,
                generation_info={"finish_reason": choice.finish_reason},
            )
        ]
    ]
    llm_output: dict[str, Any] = {}
    if getattr(response, "usage", None):
        llm_output["token_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return LLMResult(generations=generations, llm_output=llm_output)


class _OpenAIEvalLLM(BaseRagasLLM):
    def __init__(self) -> None:
        super().__init__()
        self._model = _openai_model_name(get_settings().LLM_MODEL)
        api_key = os.getenv("OPENAI_API_KEY")
        self._async_client = AsyncOpenAI(api_key=api_key)
        self._sync_client = OpenAI(api_key=api_key)
        self.set_run_config(RunConfig())

    def _get_messages(self, prompt: Any) -> list[dict[str, str]]:
        return _messages_to_openai(prompt.to_messages())

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n=n)
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=self._get_messages(prompt),
            n=n,
            temperature=temperature,
            stop=stop,
            timeout=self.run_config.timeout,
        )
        return _llm_result_from_openai(response, n)

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n=n)
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=self._get_messages(prompt),
            n=n,
            temperature=temperature,
            stop=stop,
            timeout=self.run_config.timeout,
        )
        return _llm_result_from_openai(response, n)

    def is_finished(self, response: LLMResult) -> bool:
        for gen_list in response.generations:
            resp = gen_list[0]
            finish_reason = None
            if resp.generation_info is not None:
                finish_reason = resp.generation_info.get("finish_reason")
            elif isinstance(resp, ChatGeneration):
                finish_reason = resp.message.response_metadata.get("finish_reason")
            if finish_reason not in ("stop", "STOP", "MAX_TOKENS", "eos_token", None):
                return False
        return True

    def set_run_config(self, run_config: RunConfig) -> None:
        self.run_config = run_config
        self.run_config.exception_types = RateLimitError


class _OpenAIEvalEmbeddings(BaseRagasEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self._model = "text-embedding-ada-002"
        api_key = os.getenv("OPENAI_API_KEY")
        self._async_client = AsyncOpenAI(api_key=api_key)
        self._sync_client = OpenAI(api_key=api_key)
        self.set_run_config(RunConfig())

    def embed_query(self, text: str) -> list[float]:
        response = self._sync_client.embeddings.create(
            model=self._model,
            input=text,
            timeout=self.run_config.timeout,
        )
        return cast(list[float], response.data[0].embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._sync_client.embeddings.create(
            model=self._model,
            input=texts,
            timeout=self.run_config.timeout,
        )
        return cast(list[list[float]], [d.embedding for d in response.data])

    async def aembed_query(self, text: str) -> list[float]:
        response = await self._async_client.embeddings.create(
            model=self._model,
            input=text,
            timeout=self.run_config.timeout,
        )
        return cast(list[float], response.data[0].embedding)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.embeddings.create(
            model=self._model,
            input=texts,
            timeout=self.run_config.timeout,
        )
        return cast(list[list[float]], [d.embedding for d in response.data])

    def set_run_config(self, run_config: RunConfig) -> None:
        self.run_config = run_config
        self.run_config.exception_types = RateLimitError


def get_evaluator_llm() -> BaseRagasLLM:
    return _OpenAIEvalLLM()


def get_evaluator_embeddings() -> BaseRagasEmbeddings:
    return _OpenAIEvalEmbeddings()


def run_ragas(
    dataset: EvaluationDataset, metrics: list[Any], label: str
) -> dict[str, Any]:
    from ragas import evaluate

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_evaluator_llm(),
        embeddings=get_evaluator_embeddings(),
        raise_exceptions=False,
        show_progress=False,
    )
    scores = cast(dict[str, Any], result.to_pandas().mean(numeric_only=True).to_dict())
    print(f"\n=== {label} ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    return scores
