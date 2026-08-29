from __future__ import annotations
import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        app_name: str = "Agentic RAG QA System"
        environment: str = "production"
        default_provider: str = "gemini"
        llm_model: str = "gemini-2.5-flash"
        llm_temperature: float = 0.2
        gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        chunk_size: int = 500
        chunk_overlap: int = 50
        top_k_retrieval: int = 5
        min_relevance_threshold: float = 0.60
        min_faithfulness_threshold: float = 0.70
        min_quality_threshold: float = 0.75
        max_correction_retries: int = 2

        class Config:
            env_file = ".env"
            extra = "ignore"

except Exception:
    class Settings:
        def __init__(self):
            self.app_name = "Agentic RAG QA System"
            self.environment = "production"
            self.default_provider = "gemini"
            self.llm_model = "gemini-2.5-flash"
            self.llm_temperature = 0.2
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.chunk_size = 500
            self.chunk_overlap = 50
            self.top_k_retrieval = 5
            self.min_relevance_threshold = 0.60
            self.min_faithfulness_threshold = 0.70
            self.min_quality_threshold = 0.75
            self.max_correction_retries = 2


_settings_instance = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
