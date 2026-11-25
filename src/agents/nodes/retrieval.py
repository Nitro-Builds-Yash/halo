import math
from typing import Any, Dict, List, Optional

from src.agents.state import AgentState
from src.core.config import settings
from src.core.log import get_logger
from src.rag.vectorstore import get_hybrid_retriever

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


def retrieve(state: AgentState) -> dict:
    """Hybrid (dense + BM25, RRF-fused) retrieval with cross-encoder rerank, per sub-question."""
    logger.info("retrieve: hybrid search + rerank")
    sub_questions = state.get("sub_questions") or [state["question"]]
    retriever = get_hybrid_retriever()

    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    scores: List[Optional[float]] = []
    seen_ids = set()

    for q in sub_questions:
        for item in retriever.retrieve(q, top_k=settings.TOP_K_RETRIEVE, use_rerank=True):
            content = item.get("content", "")
            meta = item.get("metadata", {}) or {}
            key = meta.get("chunk_id") or content
            if not content or key in seen_ids:
                continue
            seen_ids.add(key)
            docs.append(content)
            metas.append(meta)
            scores.append(normalize_rerank_score(item.get("rerank_score")))

    logger.info("retrieve: %d unique chunks for %d sub-question(s)", len(docs), len(sub_questions))
    return {"documents": docs, "doc_metadata": metas, "doc_scores": scores}
