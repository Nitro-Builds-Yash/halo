from __future__ import annotations
from typing import Optional
from .base_provider import BaseLLMProvider
from .mock_provider import MockLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from rag_qa_system.core.config import get_settings


class LLMProviderFactory:
    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
        settings = get_settings()
        name = (provider_name or settings.default_provider).lower()

        if name == "gemini" and settings.gemini_api_key:
            return GeminiProvider(api_key=settings.gemini_api_key, model=settings.llm_model)
        elif name in ["openai", "openrouter"] and settings.openai_api_key:
            return OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)

        return MockLLMProvider()
