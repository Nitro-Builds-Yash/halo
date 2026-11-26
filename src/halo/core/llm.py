"""Shared model clients with task routing, fallback and usage tracking.

* **Routing by task:** `task="main"` (answers, verification) uses LLM_MODEL; `task="fast"`
  (relevance grading, query planning and rewriting, follow-up condensing, multi-hop planning)
  uses FAST_LLM_MODEL when set, so cheap steps run on a cheaper model.
* **Fallback:** when FALLBACK_LLM_MODEL is set, every call that fails on the primary model
  (error, timeout, rate limit, outage) is retried once on the fallback model.
* **Usage:** every model carries a callback that records tokens and cost for the current
  question (see `halo.core.usage`).

Works with OpenAI or any OpenAI-compatible server (Ollama, vLLM, LM Studio) via the *_BASE_URL settings.
"""
from functools import lru_cache
from typing import Literal, Optional

from halo.core.config import settings

Task = Literal["main", "fast"]


def _api_key() -> str:
    # Local OpenAI-compatible servers ignore the key, but the client requires a value.
    return settings.OPENAI_API_KEY or "not-needed-for-local-server"


def model_for(task: Task) -> str:
    return (settings.FAST_LLM_MODEL if task == "fast" and settings.FAST_LLM_MODEL else settings.LLM_MODEL)


@lru_cache(maxsize=16)
def _chat_model(model: str, temperature: float, base_url: str, role: str):
    from langchain_openai import ChatOpenAI

    from halo.core.usage import UsageCallback

    kwargs = {
        "model": model,
        "temperature": temperature,
        "timeout": settings.LLM_TIMEOUT_SECONDS,
        "max_retries": settings.LLM_MAX_RETRIES,
        "api_key": _api_key(),
        "callbacks": [UsageCallback(model, role)],
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _temp(temperature: Optional[float]) -> float:
    return settings.LLM_TEMPERATURE if temperature is None else temperature


def _primary(task: Task, temperature: Optional[float]):
    return _chat_model(model_for(task), _temp(temperature), settings.LLM_BASE_URL, "primary")


def _fallback(temperature: Optional[float]):
    if not settings.FALLBACK_LLM_MODEL:
        return None
    return _chat_model(settings.FALLBACK_LLM_MODEL, _temp(temperature),
                       settings.FALLBACK_BASE_URL or settings.LLM_BASE_URL, "fallback")


def get_llm(temperature: Optional[float] = None, task: Task = "main"):
    """Chat runnable for `task`, with the fallback model attached when configured."""
    primary, fallback = _primary(task, temperature), _fallback(temperature)
    return primary.with_fallbacks([fallback]) if fallback else primary


def structured_llm(schema, temperature: Optional[float] = 0.0, task: Task = "main"):
    """Structured-output runnable for `schema`, with the fallback model attached when configured."""
    primary = _primary(task, temperature).with_structured_output(schema)
    fallback = _fallback(temperature)
    return primary.with_fallbacks([fallback.with_structured_output(schema)]) if fallback else primary


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": settings.EMBEDDING_MODEL, "api_key": _api_key(), "max_retries": settings.LLM_MAX_RETRIES}
    base_url = settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL
    if base_url:
        # Non-OpenAI models: send raw text instead of tiktoken token IDs.
        kwargs.update(base_url=base_url, check_embedding_ctx_length=False, tiktoken_enabled=False)
    return OpenAIEmbeddings(**kwargs)
