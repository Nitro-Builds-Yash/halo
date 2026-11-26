import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

from halo.core.log import get_logger

logger = get_logger(__name__)

_TOKEN_SPLIT = re.compile(r"[^\w\s]")


def _load_native():
    """The Rust extension (`pip install ./crates/halo-native`), unless disabled with HALO_NATIVE=0."""
    if os.environ.get("HALO_NATIVE", "1").lower() in ("0", "false", "no"):
        return None
    try:
        import halo_native
    except ImportError:
        return None
    return halo_native


native = _load_native()


def backend_name() -> str:
    return f"rust (halo-native {native.__version__})" if native else "python (rank-bm25)"


def tokenize(text: str) -> List[str]:
    if native:
        return native.tokenize(text)
    return _TOKEN_SPLIT.sub(" ", text.lower()).split()


def tokenize_many(texts: List[str]) -> List[List[str]]:
    if native:
        return native.tokenize_many(list(texts))
    return [tokenize(t) for t in texts]


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class BM25Index:
    """Persistent sparse BM25 index for exact keyword matching.

    Chunks carry the same `chunk_id` as in Chroma so dense and sparse results can be
    fused by ID. BM25Plus avoids negative/zero IDF on small corpora.

    Scoring runs in Rust when the optional `halo_native` extension is installed, and in
    pure Python (rank-bm25) otherwise; both give the same scores and the same ranking.
    """

    def __init__(self, persist_path: str = "./chroma_db/bm25_index.json"):
        self.persist_path = persist_path
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25 = None
        self._lock = threading.RLock()
        self._mtime: Optional[float] = None
        self._load()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return tokenize(text)

    def _rebuild(self):
        if not self.corpus_tokens:
            self.bm25 = None
        elif native:
            self.bm25 = native.Bm25Index(self.corpus_tokens)
        else:
            from rank_bm25 import BM25Plus

            self.bm25 = BM25Plus(self.corpus_tokens)

    def _file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.persist_path)
        except OSError:
            return None

    def _refresh(self):
        """Reload if another process (e.g. the API vs. the Streamlit UI) changed the index file."""
        mtime = self._file_mtime()
        if mtime is not None and mtime != self._mtime:
            self._load()

    def __len__(self) -> int:
        with self._lock:
            self._refresh()
            return len(self.documents)

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None):
        if not documents:
            return
        metadatas = metadatas or [{} for _ in documents]
        ids = ids or [m.get("chunk_id") or content_id(d) for d, m in zip(documents, metadatas)]
        with self._lock:
            self._refresh()
            existing = set(self.ids)
            for cid, doc, meta in zip(ids, documents, metadatas):
                if cid in existing:  # idempotent: re-adding the same chunk is a no-op
                    continue
                existing.add(cid)
                self.ids.append(cid)
                self.documents.append(doc)
                self.metadatas.append({**meta, "chunk_id": cid})
                self.corpus_tokens.append(self._tokenize(doc))
            self._rebuild()
            self._save()

    def sources(self) -> List[Dict[str, Any]]:
        """One entry per indexed source: chunk count and pages covered."""
        with self._lock:
            self._refresh()
            summary: Dict[str, Dict[str, Any]] = {}
            for m in self.metadatas:
                src = m.get("source", "unknown")
                entry = summary.setdefault(src, {"source": src, "chunks": 0, "pages": set()})
                entry["chunks"] += 1
                if m.get("page") is not None:
                    entry["pages"].add(m["page"])
            return [{"source": e["source"], "chunks": e["chunks"], "pages": len(e["pages"])}
                    for e in sorted(summary.values(), key=lambda e: e["source"].lower())]

    def delete_source(self, source: str) -> int:
        """Remove every chunk whose metadata `source` matches. Returns the number removed."""
        with self._lock:
            self._refresh()
            keep = [i for i, m in enumerate(self.metadatas) if m.get("source") != source]
            removed = len(self.ids) - len(keep)
            if removed:
                self.ids = [self.ids[i] for i in keep]
                self.documents = [self.documents[i] for i in keep]
                self.metadatas = [self.metadatas[i] for i in keep]
                self.corpus_tokens = [self.corpus_tokens[i] for i in keep]
                self._rebuild()
                self._save()
            return removed

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            if not self.bm25 or not self.documents:
                return []
            tokens = self._tokenize(query)
            if not tokens:
                return []
            if native:
                ranked = self.bm25.top_k(tokens, top_k)
            else:
                scores = self.bm25.get_scores(tokens)
                query_set = set(tokens)
                ranked = []
                for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
                    if query_set.intersection(self.corpus_tokens[idx]):
                        ranked.append((idx, scores[idx]))
                    if len(ranked) >= top_k:
                        break
            results = [{
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(score),
            } for idx, score in ranked]
            return results

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        tmp = self.persist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}, f, ensure_ascii=False)
        os.replace(tmp, self.persist_path)  # atomic write
        self._mtime = self._file_mtime()

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        self.ids, self.documents, self.metadatas, self.corpus_tokens = [], [], [], []
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", []) or [{} for _ in self.documents]
            # Older index files have no ids: derive them from the content.
            self.ids = data.get("ids") or [m.get("chunk_id") or content_id(d)
                                           for d, m in zip(self.documents, self.metadatas)]
            self.corpus_tokens = tokenize_many(self.documents)
            self._rebuild()
            self._mtime = self._file_mtime()
        except Exception:
            logger.exception("Failed to load BM25 index from %s", self.persist_path)
