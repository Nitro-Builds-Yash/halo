from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # LLM / Embedding Settings
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Point at any OpenAI-compatible server (e.g. Ollama: http://localhost:11434/v1) to run without OpenAI.
    LLM_BASE_URL: str = ""
    EMBEDDING_BASE_URL: str = ""
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 2

    # RAG Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"   # local mode (single process); also holds BM25 + cache files
    CHROMA_HOST: str = ""                       # set to use a Chroma server (needed when API and UI run separately)
    CHROMA_PORT: int = 8000
    COLLECTION_NAME: str = "halo_documents"
    TOP_K_RETRIEVE: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_CHARS: int = 24000   # total context sent to the generator / graders
    MAX_DOC_CHARS: int = 4000        # per document (long web pages are truncated)

    # Self-correction ceilings
    MAX_REWRITES: int = 1          # query rewrites + web searches per question
    MAX_REGENERATIONS: int = 1     # retries after a failed grounding/answer check

    # Cross-encoder gatekeeper thresholds (sigmoid-normalised scores, 0..1)
    GATE_HIGH: float = 0.85        # chunks at/above this skip LLM grading
    GATE_LOW: float = 0.40         # chunks below this are discarded

    # Semantic cache
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY: float = 0.92
    CACHE_TTL_SECONDS: int = 86400
    CACHE_MIN_CONFIDENCE: float = 0.70

    # Server / App Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: str = ""              # if set, /api/v1/* requires the X-API-Key header
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"
    MAX_UPLOAD_MB: int = 25
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def llm_configured(self) -> bool:
        """True if an LLM/embedding backend is reachable without further setup."""
        return bool(self.OPENAI_API_KEY or self.LLM_BASE_URL)

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
