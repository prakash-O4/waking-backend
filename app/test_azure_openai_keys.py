"""Smoke test: verify AZURE_OPENAI_LLM_KEY and AZURE_OPENAI_LLM_ENDPOINT are valid."""
import os
import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


@pytest.fixture(scope="module")
def azure_client():
    key = os.getenv("AZURE_OPENAI_LLM_KEY")
    endpoint_full = os.getenv("AZURE_OPENAI_LLM_ENDPOINT", "")
    assert key, "AZURE_OPENAI_LLM_KEY is not set in .env"
    assert endpoint_full, "AZURE_OPENAI_LLM_ENDPOINT is not set in .env"

    base_url = endpoint_full.split("/openai/")[0] if "/openai/" in endpoint_full else endpoint_full
    return AzureOpenAI(
        api_key=key,
        api_version="2025-01-01-preview",
        azure_endpoint=base_url,
    )


def test_azure_openai_chat_completion(azure_client):
    response = azure_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        max_tokens=5,
    )
    assert response.choices, "No choices returned from Azure OpenAI"
    content = response.choices[0].message.content.strip()
    assert content, "Empty response from Azure OpenAI"
    print(f"\nAzure OpenAI responded: {content!r}")
