from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    
    # LLM Settings
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    
    # RAG Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "halo_documents"
    TOP_K_RETRIEVE: int = 4
    MAX_RETRIES: int = 3
    
    # Server / App Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
