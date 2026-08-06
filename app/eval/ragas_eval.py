from __future__ import annotations

from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from app.config import get_settings


def get_evaluator_llm() -> LangchainLLMWrapper:
    settings = get_settings()
    return LangchainLLMWrapper(init_chat_model(settings.LLM_MODEL))


def get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def run_ragas(
    dataset: EvaluationDataset, metrics: list[Any], label: str
) -> dict[str, Any]:
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
