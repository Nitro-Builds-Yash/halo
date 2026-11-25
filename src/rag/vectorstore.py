import os
import threading
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.llm import get_embeddings
from src.core.log import get_logger
from src.rag.bm25 import BM25Index
from src.rag.reranker import CrossEncoderReranker

logger = get_logger(__name__)

_lock = threading.Lock()
_bm25_index: Optional[BM25Index] = None
_vectorstore = None
_hybrid: Optional["HybridRetriever"] = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    with _lock:
        if _bm25_index is None:
            _bm25_index = BM25Index(persist_path=os.path.join(settings.CHROMA_PERSIST_DIR, "bm25_index.json"))
        return _bm25_index


def get_vectorstore():
    """Single shared Chroma client (previously a new one was built on every query)."""
    global _vectorstore
    with _lock:
        if _vectorstore is None:
            from langchain_chroma import Chroma

            _vectorstore = Chroma(
                collection_name=settings.COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=settings.CHROMA_PERSIST_DIR,
            )
        return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.TOP_K_RETRIEVE})


def get_hybrid_retriever() -> "HybridRetriever":
    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever()
    return _hybrid


class HybridRetriever:
    """Dense (Chroma) + sparse (BM25) retrieval fused with Reciprocal Rank Fusion,
    then reranked with a cross-encoder. Results are fused on stable chunk IDs."""

    def __init__(self, rrf_k: int = 60, vectorstore=None, bm25: Optional[BM25Index] = None, reranker=None):
        self.vectorstore = vectorstore if vectorstore is not None else get_vectorstore()
        self.bm25 = bm25 if bm25 is not None else get_bm25_index()
        self.reranker = reranker if reranker is not None else CrossEncoderReranker.get_instance()
        self.rrf_k = rrf_k

    @staticmethod
    def _key(item: Dict[str, Any]) -> str:
        return item.get("id") or item.get("metadata", {}).get("chunk_id") or item["content"]

    def _dense(self, query: str, k: int) -> List[Dict[str, Any]]:
        try:
            pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            logger.exception("Dense search failed")
            return []
        return [{
            "id": doc.metadata.get("chunk_id") or getattr(doc, "id", None),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "dense_score": float(score),
        } for doc, score in pairs]

    def retrieve(self, query: str, top_k: int = 4, use_rerank: bool = True) -> List[Dict[str, Any]]:
        dense = self._dense(query, top_k * 2)
        sparse = self.bm25.search(query, top_k=top_k * 2)

        rrf: Dict[str, float] = {}
        lookup: Dict[str, Dict[str, Any]] = {}
        for results in (dense, sparse):
            for rank, item in enumerate(results):
                key = self._key(item)
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                lookup.setdefault(key, item)

        fused = []
        for key, score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
            fused.append({**lookup[key], "rrf_score": score})

        if use_rerank and fused:
            return self.reranker.rerank(query, fused, top_k=top_k)
        return fused[:top_k]
