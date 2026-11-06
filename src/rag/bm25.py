import os
import json
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Plus, BM25Okapi

class BM25Index:
    """
    Persistent Sparse BM25 lexical search index for exact keyword matching.
    Uses BM25Plus to avoid negative/zero IDF on small corpora.
    """
    def __init__(self, persist_path: str = "./chroma_db/bm25_index.json"):
        self.persist_path = persist_path
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return cleaned.split()

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if not documents:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
            
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        new_tokens = [self._tokenize(doc) for doc in documents]
        self.corpus_tokens.extend(new_tokens)
        
        if self.corpus_tokens:
            self.bm25 = BM25Plus(self.corpus_tokens)
            
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []
            
        tokens = self._tokenize(query)
        if not tokens:
            return []
            
        scores = self.bm25.get_scores(tokens)
        # Filter for documents containing at least one query term
        query_set = set(tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in scored_indices:
            # Verify term overlap
            doc_tokens = set(self.corpus_tokens[idx])
            overlap = query_set.intersection(doc_tokens)
            if overlap:
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                    "score": float(scores[idx]),
                    "index": idx
                })
            if len(results) >= top_k:
                break

        return results

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        data = {
            "documents": self.documents,
            "metadatas": self.metadatas
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
                self.corpus_tokens = [self._tokenize(doc) for doc in self.documents]
                if self.corpus_tokens:
                    self.bm25 = BM25Plus(self.corpus_tokens)
            except Exception as e:
                print(f"Warning: Failed to load BM25 index from {self.persist_path}: {e}")
