import threading
from typing import Any, Dict, List

from src.core.log import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).

    `rerank_score` is the raw logit; callers normalise it with a sigmoid. If
    sentence-transformers/torch are not installed, results keep their RRF order and
    carry no `rerank_score` (downstream, every chunk is then graded by the LLM).
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_model(self):
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            try:
                import torch
                from sentence_transformers import CrossEncoder

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CrossEncoder(self.model_name, device=device)
            except Exception as e:
                logger.warning("Cross-encoder unavailable, reranking disabled: %s", e)
                self._model = None

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._model is not None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
        if not self.available:
            return documents[:top_k]
        try:
            scores = self._model.predict([(query, d.get("content", "")) for d in documents])
            reranked = [{**d, "rerank_score": float(s)} for d, s in zip(documents, scores)]
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception:
            logger.exception("Reranking failed; using fused order")
            return documents[:top_k]
