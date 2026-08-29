from __future__ import annotations
from typing import List, Dict
from rag_qa_system.core.schemas import DocumentChunk
from .bm25_store import BM25Store
from .vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 keyword search + Vector dense search via RRF.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25_store = BM25Store()
        self.vector_store = VectorStore()
        self.is_indexed = False

    def index_documents(self, chunks: List[DocumentChunk]):
        self.bm25_store.index_chunks(chunks)
        self.vector_store.index_chunks(chunks)
        self.is_indexed = True

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        if not self.is_indexed:
            return []

        bm25_results = self.bm25_store.search(query, top_k=top_k * 2)
        vector_results = self.vector_store.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(vector_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            chunk_map[chunk.chunk_id] = chunk

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results: List[DocumentChunk] = []

        for chunk_id, score in sorted_chunks[:top_k]:
            c = chunk_map[chunk_id].copy()
            c.score = float(score)
            final_results.append(c)

        return final_results
