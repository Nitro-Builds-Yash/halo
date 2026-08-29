from __future__ import annotations
import math
import re
from typing import List, Dict, Tuple
from rag_qa_system.core.schemas import DocumentChunk


class VectorStore:
    """High-performance semantic vector store with cosine similarity ranking."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[int, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        n = len(chunks)
        if n == 0:
            return

        doc_tokens = [self._tokenize(c.content) for c in self.chunks]
        vocab = {}
        doc_counts = {}
        for tokens in doc_tokens:
            seen = set(tokens)
            for t in seen:
                doc_counts[t] = doc_counts.get(t, 0) + 1
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        self.idf = {t: math.log((n + 1) / (count + 1)) + 1.0 for t, count in doc_counts.items()}

        self.vectors = []
        for tokens in doc_tokens:
            vec = self._vectorize(tokens)
            self.vectors.append(vec)

    def _vectorize(self, tokens: List[str]) -> Dict[int, float]:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        norm_sq = 0.0
        for t, count in tf.items():
            if t in self.vocab:
                dim = self.vocab[t]
                w = count * self.idf.get(t, 1.0)
                vec[dim] = w
                norm_sq += w * w
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for dim in vec:
                vec[dim] /= norm
        return vec

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        if not self.chunks or not query:
            return []

        q_tokens = self._tokenize(query)
        q_vec = self._vectorize(q_tokens)
        if not q_vec:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.vectors):
            dot_product = sum(weight * doc_vec.get(dim, 0.0) for dim, weight in q_vec.items())
            if dot_product > 0:
                scores.append((idx, dot_product))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[DocumentChunk] = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx].copy()
            c.score = float(score)
            results.append(c)
        return results
