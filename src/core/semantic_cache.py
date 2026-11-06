import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

class SemanticCache:
    """
    Lightweight entry-point semantic cache for FastAPI routes.
    Checks if a semantically identical query (cosine similarity > threshold, default 0.92)
    has been verified and answered recently.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cache_file = os.path.join(settings.CHROMA_PERSIST_DIR, "semantic_cache.json")
            cls._instance = cls(cache_path=cache_file, similarity_threshold=0.92)
        return cls._instance

    def __init__(self, cache_path: str = "./chroma_db/semantic_cache.json", similarity_threshold: float = 0.92, ttl_seconds: int = 86400):
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {} # id -> {query, embedding, response, timestamp}
        self.embedder = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else "dummy-key")
        self._load()

    def _cosine_similarity(self, v1: list, v2: list) -> float:
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def lookup(self, query: str) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Returns cached response and matched similarity if cosine similarity > threshold, else None.
        """
        if not self.cache or not settings.OPENAI_API_KEY:
            return None

        try:
            query_vec = self.embedder.embed_query(query)
            now = time.time()
            best_match = None
            highest_sim = -1.0

            for entry_id, entry in list(self.cache.items()):
                # Expiry check
                if now - entry.get("timestamp", 0) > self.ttl_seconds:
                    del self.cache[entry_id]
                    continue

                sim = self._cosine_similarity(query_vec, entry["embedding"])
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = entry

            if highest_sim >= self.similarity_threshold and best_match:
                return best_match["response"], highest_sim
        except Exception as e:
            print(f"SemanticCache lookup notice: {e}")

        return None

    def store(self, query: str, response: Dict[str, Any]):
        """
        Stores query embedding, verified answer, and citations in the persistent cache.
        """
        if not settings.OPENAI_API_KEY:
            return

        try:
            query_vec = self.embedder.embed_query(query)
            entry_id = str(abs(hash(query.strip().lower())))
            self.cache[entry_id] = {
                "query": query,
                "embedding": query_vec,
                "response": response,
                "timestamp": time.time()
            }
            self._save()
        except Exception as e:
            print(f"SemanticCache store notice: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"SemanticCache save notice: {e}")

    def _load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"SemanticCache load notice: {e}")
                self.cache = {}
