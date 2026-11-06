from typing import List, Dict, Any
import os

class CrossEncoderReranker:
    """
    Neural Cross-Encoder reranker using ms-marco-MiniLM-L-6-v2.
    Computes cross-attention between (query, document) pairs for superior semantic precision.
    Falls back gracefully if torch or weights are not available.
    """
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._initialized = False

    def _ensure_model(self):
        if not self._initialized:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CrossEncoder(self.model_name, device=device)
                self._initialized = True
            except Exception as e:
                print(f"Notice: CrossEncoder initialization deferred or running in fallback mode: {e}")
                self._model = None
                self._initialized = True

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        self._ensure_model()
        if self._model is None:
            # Fallback to current score order
            return documents[:top_k]

        try:
            pairs = [(query, doc.get("content", "")) for doc in documents]
            scores = self._model.predict(pairs)

            reranked = []
            for doc, score in zip(documents, scores):
                item = dict(doc)
                item["rerank_score"] = float(score)
                reranked.append(item)

            reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked[:top_k]
        except Exception as e:
            print(f"Warning: Reranking failed, using original ranking: {e}")
            return documents[:top_k]
