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

    # Routing and fallback
    FAST_LLM_MODEL: str = ""       # cheaper model for grading, planning, rewriting (default: LLM_MODEL)
    FALLBACK_LLM_MODEL: str = ""   # retried when the primary model errors or times out
    FALLBACK_BASE_URL: str = ""    # e.g. a local Ollama server as the fallback (default: LLM_BASE_URL)

    # Cost tracking
    MODEL_PRICES: str = ""         # JSON {"model": [input_usd_per_1M, output_usd_per_1M]} to add/override prices
    MAX_COST_PER_QUERY_USD: float = 0.0  # stop a question once it has spent this much (0 = no limit)

    # Prompt-injection guard: "sanitize" neutralizes injected instructions, "flag" only reports, "off"
    INJECTION_GUARD: str = "sanitize"

    # Agentic multi-hop search: "auto" (multi-part questions), "on" (every question) or "off"
    AGENTIC_SEARCH: str = "auto"
    MAX_HOPS: int = 2

    # RAG Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"   # local mode (single process); also holds BM25 + cache files
    CHROMA_HOST: str = ""                       # set to use a Chroma server (needed when API and UI run separately)
    CHROMA_PORT: int = 8000
    COLLECTION_NAME: str = "halo_documents"
    TOP_K_RETRIEVE: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DOCUMENTS_DIR: str = ""        # where original uploads are kept for citation links (default: <CHROMA_PERSIST_DIR>/files)
    MAX_CONTEXT_CHARS: int = 24000   # total context sent to the generator / graders
    MAX_DOC_CHARS: int = 4000        # per document (long web pages are truncated)

    # Document extraction
    EXTRACT_TABLES: bool = True      # PDF tables become their own Markdown-table chunks
    OCR: str = "auto"                # "auto": OCR scanned pages and images when Tesseract is installed; "off"
    OCR_LANG: str = "eng"            # Tesseract language(s), e.g. "eng+hin"
    OCR_MIN_CHARS: int = 25          # PDF pages with less extractable text than this are OCR'd
    OCR_TIMEOUT_SECONDS: int = 120
    TESSERACT_CMD: str = ""          # path to the tesseract binary if it isn't on PATH

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

    # Optional LangSmith tracing
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "halo"
    LANGSMITH_ENDPOINT: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def documents_dir(self) -> str:
        import os

        return self.DOCUMENTS_DIR or os.path.join(self.CHROMA_PERSIST_DIR, "files")

    @property
    def llm_configured(self) -> bool:
        """True if an LLM/embedding backend is reachable without further setup."""
        return bool(self.OPENAI_API_KEY or self.LLM_BASE_URL)

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
