from typing import Any, Dict, List

from halo.agents.nodes.generation import NO_CONTEXT_ANSWER
from halo.agents.state import AgentState
from halo.core.log import get_logger

logger = get_logger(__name__)


def graceful_fallback(state: AgentState) -> dict:
    """Deterministic fallback once self-correction is exhausted: return the best excerpts
    with a clear disclaimer instead of an unverified answer."""
    documents = state.get("documents", [])
    metas = state.get("doc_metadata", [])
    logger.info("fallback: verdict=%s, %d document(s)", state.get("verdict"), len(documents))

    if not documents:
        return {"generation": NO_CONTEXT_ANSWER, "confidence": 0.0, "citations": [], "sentences": [], "is_fallback": True}

    citations: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents[:3]):
        meta = metas[idx] if idx < len(metas) else {}
        citations.append({
            "source": meta.get("source", "Indexed Document"),
            "page": meta.get("page"),
            "snippet": doc[:300] + ("..." if len(doc) > 300 else ""),
        })

    attempts = 1 + state.get("retry_count", 0)
    text = (
        f"**Partial confidence:** I couldn't produce a fully verified answer after {attempts} attempt(s). "
        "These are the most relevant excerpts I found; please check the cited sources directly."
    )
    return {"generation": text, "confidence": 0.3, "citations": citations, "sentences": [], "is_fallback": True}
