from __future__ import annotations
import math
import re
from typing import List, Dict, Tuple
from rag_qa_system.core.schemas import DocumentChunk


class BM25Store:
    """In-memory BM25 Okapi lexical index for exact keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.corpus_size: int = 0
        self.tf_list: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.tf_list = []
        self.doc_freqs = {}

        total_len = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.tf_list.append(tf)

            for token in tf.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        if not self.chunks or not query:
            return []

        q_tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []

        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf_dict = self.tf_list[idx]

            for token in q_tokens:
                if token not in tf_dict:
                    continue
                tf = tf_dict[token]
                df = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (num / den)

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[DocumentChunk] = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx].copy()
            c.score = float(score)
            results.append(c)
        return results
