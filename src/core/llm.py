"""Shared, lazily-created model clients so nodes don't rebuild them on every call.

Works with OpenAI or any OpenAI-compatible server (Ollama, vLLM, LM Studio) via
LLM_BASE_URL / EMBEDDING_BASE_URL.
"""
from functools import lru_cache

from src.core.config import settings


def _api_key() -> str:
    # Local OpenAI-compatible servers ignore the key, but the client requires a value.
    return settings.OPENAI_API_KEY or "not-needed-for-local-server"


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
        "timeout": settings.LLM_TIMEOUT_SECONDS,
        "max_retries": settings.LLM_MAX_RETRIES,
        "api_key": _api_key(),
    }
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": settings.EMBEDDING_MODEL, "api_key": _api_key(), "max_retries": settings.LLM_MAX_RETRIES}
    base_url = settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL
    if base_url:
        # Non-OpenAI models: send raw text instead of tiktoken token IDs.
        kwargs.update(base_url=base_url, check_embedding_ctx_length=False, tiktoken_enabled=False)
    return OpenAIEmbeddings(**kwargs)
