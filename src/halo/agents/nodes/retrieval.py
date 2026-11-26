import math
from typing import Any, Dict, List, Optional

from halo.agents.state import AgentState
from halo.core.config import settings
from halo.core.guard import inspect
from halo.core.log import get_logger
from halo.rag.vectorstore import get_hybrid_retriever

logger = get_logger(__name__)


def normalize_rerank_score(score: Optional[float]) -> Optional[float]:
    """Map a cross-encoder logit to 0..1 with a sigmoid.

    ms-marco cross-encoders return raw logits, so the sigmoid is applied to every
    score. (Applying it only to values outside 0..1 made the mapping non-monotonic:
    a logit of 0.9 stayed 0.9 while a better logit of 1.1 became 0.75.)
    """
    if score is None:
        return None
    try:
        return 1.0 / (1.0 + math.exp(-float(score)))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


class Collector:
    """Accumulates unique, guard-checked chunks across several searches."""

    def __init__(self, documents=None, metas=None, scores=None, flags=None):
        self.docs: List[str] = list(documents or [])
        self.metas: List[Dict[str, Any]] = list(metas or [])
        self.scores: List[Optional[float]] = list(scores or [None] * len(self.docs))
        self.flags: List[Dict[str, Any]] = list(flags or [])
        self.seen = {m.get("chunk_id") or d for d, m in zip(self.docs, self.metas)}

    def add(self, items) -> int:
        added = 0
        for item in items:
            content = item.get("content", "")
            meta = item.get("metadata", {}) or {}
            key = meta.get("chunk_id") or content
            if not content or key in self.seen:
                continue
            self.seen.add(key)
            content, flags = inspect(content, meta)
            self.flags += flags
            self.docs.append(content)
            self.metas.append(meta)
            self.scores.append(normalize_rerank_score(item.get("rerank_score")))
            added += 1
        return added

    def state(self) -> dict:
        return {"documents": self.docs, "doc_metadata": self.metas, "doc_scores": self.scores,
                "security_flags": self.flags}


def retrieve(state: AgentState) -> dict:
    """Hybrid (dense + BM25, RRF-fused) retrieval with cross-encoder rerank, per sub-question."""
    logger.info("retrieve: hybrid search + rerank")
    sub_questions = state.get("sub_questions") or [state["question"]]
    retriever = get_hybrid_retriever()

    collector = Collector(flags=state.get("security_flags"))
    for q in sub_questions:
        collector.add(retriever.retrieve(q, top_k=settings.TOP_K_RETRIEVE, use_rerank=True))

    logger.info("retrieve: %d unique chunks for %d sub-question(s)", len(collector.docs), len(sub_questions))
    return collector.state()
