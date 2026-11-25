"""Single entry point for running a question through the pipeline (used by the API, CLI and Streamlit)."""
from typing import Any, Dict, Optional, Tuple

from halo.agents.state import initial_state
from halo.core.config import settings
from halo.core.semantic_cache import SemanticCache


def used_web_search(state: Dict[str, Any]) -> bool:
    return any(m.get("type") == "web" for m in state.get("doc_metadata", []) or [])


def summarize(question: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": question,
        "generation": state.get("generation", ""),
        "confidence": float(state.get("confidence", 0.0) or 0.0),
        "citations": state.get("citations", []),
        "documents": state.get("documents", []),
        "verdict": state.get("verdict"),
        "retry_count": state.get("retry_count", 0),
        "rewrite_count": state.get("rewrite_count", 0),
        "web_search_used": used_web_search(state),
        "is_fallback": state.get("is_fallback", False),
        "cached": False,
        "cached_similarity": None,
    }


def should_cache(result: Dict[str, Any]) -> bool:
    return (
        result.get("verdict") == "useful"
        and not result.get("is_fallback", False)
        and result.get("confidence", 0.0) >= settings.CACHE_MIN_CONFIDENCE
    )


def cached_answer(question: str, cache: SemanticCache) -> Optional[Dict[str, Any]]:
    hit: Optional[Tuple[Dict[str, Any], float]] = cache.lookup(question)
    if not hit:
        return None
    data, sim = hit
    return {**summarize(question, data), "verdict": "useful", "cached": True, "cached_similarity": round(sim, 4)}


def cache_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("generation", "confidence", "citations", "documents", "doc_metadata")
    return {k: result[k] for k in keys if k in result}


def run_query(question: str, graph=None, cache: Optional[SemanticCache] = None) -> Dict[str, Any]:
    """Synchronous run with the semantic cache in front (CLI / Streamlit)."""
    if graph is None:
        from halo.agents.graph import graph as default_graph

        graph = default_graph
    cache = cache or SemanticCache.get_instance()

    hit = cached_answer(question, cache)
    if hit:
        return hit

    state = graph.invoke(initial_state(question))
    result = summarize(question, state)
    if should_cache(result):
        cache.store(question, cache_payload(state))
    return result
