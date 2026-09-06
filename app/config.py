# config.py
from __future__ import annotations

from pydantic_settings import BaseSettings
from functools import lru_cache
import secrets
import base64
import os

def generate_api_key(length: int = 32) -> str:
    """
    Generates a secure API key using cryptographically strong random bytes.
    Returns a URL-safe base64-encoded string.
    """
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')

def generate_api_secret(length: int = 64) -> str:
    """
    Generates a secure API secret using cryptographically strong random bytes.
    Returns a hex string.
    """
    return secrets.token_hex(length)

class Settings(BaseSettings):
    API_KEY: str = ""
    API_SECRET: str = ""
    LOG_LEVEL: str = "INFO"
    LLM_MODEL: str = "openai:gpt-4o-mini"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_LOG_CONTENT: bool = False
    # Embedding (text-embedding-3-large)
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""  # full deployment URL or base URL
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"
    AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = 1024
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    COHERE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    # LLM (gpt-4.1-mini)
    AZURE_OPENAI_LLM_KEY: str = ""
    AZURE_OPENAI_LLM_ENDPOINT: str = ""  # full deployment URL or base URL
    AZURE_OPENAI_LLM_DEPLOYMENT: str = "gpt-4.1-mini"
    AZURE_OPENAI_LLM_API_VERSION: str = "2024-10-21"

    class Config:
        env_file = ".env"
        # Tolerate unrelated keys in .env (e.g. legacy pinecone/supabase entries);
        # without this, Settings() raises on any extra key and nothing boots.
        extra = "ignore"


def azure_base_url(endpoint: str | None = None) -> str:
    """Strip deployment path from an Azure OpenAI endpoint URL, returning the base URL.

    If no endpoint is provided, falls back to AZURE_OPENAI_ENDPOINT from settings.
    """
    url = endpoint if endpoint is not None else get_settings().AZURE_OPENAI_ENDPOINT
    if "/openai/deployments/" in url:
        return url.split("/openai/deployments/")[0].rstrip("/")
    return url.rstrip("/")

def initialize_keys():
    """
    Initializes API keys if they don't exist in the .env file.
    Returns a tuple of (api_key, api_secret).
    """
    env_path = '.env'
    
    # Read existing environment variables
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value

    # Generate new keys if they don't exist
    api_key = env_vars.get('API_KEY', '')
    api_secret = env_vars.get('API_SECRET', '')
    
    updated = False
    
    if not api_key:
        api_key = generate_api_key()
        env_vars['API_KEY'] = api_key
        updated = True
        
    if not api_secret:
        api_secret = generate_api_secret()
        env_vars['API_SECRET'] = api_secret
        updated = True

    # Write back to .env if updates were made
    if updated:
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
    
    return api_key, api_secret

@lru_cache()
def get_settings():
    # Initialize keys before creating settings
    initialize_keys()
    return Settings()
