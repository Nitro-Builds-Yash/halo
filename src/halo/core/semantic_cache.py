import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from halo.core.config import settings
from halo.core.log import get_logger

logger = get_logger(__name__)


def cache_key(query: str) -> str:
    # hashlib, not hash(): Python's hash() is randomised per process, so keys changed on every restart.
    return hashlib.sha256(" ".join(query.lower().split()).encode("utf-8")).hexdigest()


class SemanticCache:
    """Entry-point cache of verified answers, matched by cosine similarity of query embeddings.

    Entries are cleared whenever documents are (re)ingested, so stale answers aren't served.
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    cache_path=os.path.join(settings.CHROMA_PERSIST_DIR, "semantic_cache.json"),
                    similarity_threshold=settings.CACHE_SIMILARITY,
                    ttl_seconds=settings.CACHE_TTL_SECONDS,
                )
            return cls._instance

    def __init__(self, cache_path: str = "./chroma_db/semantic_cache.json", similarity_threshold: float = 0.92,
                 ttl_seconds: int = 86400, embedder=None, enabled: Optional[bool] = None):
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.enabled = settings.CACHE_ENABLED if enabled is None else enabled
        self._embedder = embedder
        self._lock = threading.RLock()
        self._recent_vecs: "OrderedDict[str, List[float]]" = OrderedDict()  # avoid embedding twice per request
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._mtime: Optional[float] = None
        self._load()

    @property
    def embedder(self):
        if self._embedder is None:
            from halo.core.llm import get_embeddings

            self._embedder = get_embeddings()
        return self._embedder

    def _can_embed(self) -> bool:
        return self._embedder is not None or settings.llm_configured

    def _embed(self, query: str) -> List[float]:
        key = cache_key(query)
        if key in self._recent_vecs:
            return self._recent_vecs[key]
        vec = self.embedder.embed_query(query)
        self._recent_vecs[key] = vec
        if len(self._recent_vecs) > 256:
            self._recent_vecs.popitem(last=False)
        return vec

    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        a, b = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, e in self.cache.items() if now - e.get("timestamp", 0) > self.ttl_seconds]
        for k in expired:
            del self.cache[k]

    def lookup(self, query: str) -> Optional[Tuple[Dict[str, Any], float]]:
        if not self.enabled or not self._can_embed():
            return None
        with self._lock:
            self._refresh()
            self._evict_expired(time.time())
            if not self.cache:
                return None
            entries = list(self.cache.values())
        try:
            q = np.asarray(self._embed(query), dtype=float)
            mat = np.asarray([e["embedding"] for e in entries], dtype=float)
            denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
            sims = (mat @ q) / np.where(denom == 0, 1.0, denom)
            best = int(np.argmax(sims))
            if sims[best] >= self.similarity_threshold:
                return entries[best]["response"], float(sims[best])
        except Exception:
            logger.exception("Semantic cache lookup failed")
        return None

    def store(self, query: str, response: Dict[str, Any]) -> None:
        if not self.enabled or not self._can_embed():
            return
        try:
            vec = self._embed(query)
            with self._lock:
                self._refresh()
                self.cache[cache_key(query)] = {
                    "query": query,
                    "embedding": vec,
                    "response": response,
                    "timestamp": time.time(),
                }
                self._save()
        except Exception:
            logger.exception("Semantic cache store failed")

    def clear(self) -> None:
        with self._lock:
            self.cache = {}
            self._save()
        logger.info("Semantic cache cleared")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
            os.replace(tmp, self.cache_path)
            self._mtime = self._file_mtime()
        except Exception:
            logger.exception("Semantic cache save failed")

    def _file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.cache_path)
        except OSError:
            return None

    def _refresh(self) -> None:
        """Pick up changes made by another process (e.g. a clear() after ingestion in the API)."""
        mtime = self._file_mtime()
        if mtime is not None and mtime != self._mtime:
            self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                self._mtime = self._file_mtime()
            except Exception:
                logger.exception("Semantic cache load failed; starting empty")
                self.cache = {}
