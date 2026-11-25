"""Shared, lazily-created model clients so nodes don't rebuild them on every call."""
from functools import lru_cache

from src.core.config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    from langchain_openai import ChatOpenAI

    temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    return ChatOpenAI(model=settings.LLM_MODEL, temperature=temp)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    api_key = settings.OPENAI_API_KEY or "missing-openai-key"
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=api_key)
