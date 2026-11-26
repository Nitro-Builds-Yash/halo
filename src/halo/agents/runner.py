"""Single entry point for running a question through the pipeline (used by the API, CLI and Streamlit)."""
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from halo.agents.conversation import condense_question
from halo.agents.state import initial_state
from halo.core.config import settings
from halo.core.semantic_cache import SemanticCache
from halo.core.tracing import run_config


def used_web_search(state: Dict[str, Any]) -> bool:
    return any(m.get("type") == "web" for m in state.get("doc_metadata", []) or [])


def link_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add a `url` to each citation: the web page for web results, or the API route that serves
    the stored document (opened at the cited page for PDFs). Relative to the API's base URL."""
    linked = []
    for c in citations or []:
        c = dict(c)
        source = str(c.get("source") or "")
        if source.startswith(("http://", "https://")):
            c["url"] = source
        elif source and os.path.isfile(os.path.join(settings.documents_dir, os.path.basename(source))):
            url = f"/api/v1/documents/{quote(os.path.basename(source))}/file"
            if source.lower().endswith(".pdf") and c.get("page"):
                url += f"#page={c['page']}"
            c["url"] = url
        linked.append(c)
    return linked


def summarize(question: str, state: Dict[str, Any], standalone_question: Optional[str] = None) -> Dict[str, Any]:
    return {
        "question": question,
        "standalone_question": standalone_question or state.get("question") or question,
        "generation": state.get("generation", ""),
        "confidence": float(state.get("confidence", 0.0) or 0.0),
        "citations": link_citations(state.get("citations", [])),
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


def cached_answer(question: str, cache: SemanticCache, standalone_question: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up the (standalone) question in the cache. `question` is what the user typed."""
    key = standalone_question or question
    hit: Optional[Tuple[Dict[str, Any], float]] = cache.lookup(key)
    if not hit:
        return None
    data, sim = hit
    return {**summarize(question, data, key), "verdict": "useful", "cached": True, "cached_similarity": round(sim, 4)}


def cache_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("generation", "confidence", "citations", "documents", "doc_metadata")
    return {k: result[k] for k in keys if k in result}


def run_query(question: str, graph=None, cache: Optional[SemanticCache] = None,
              history: Optional[List[Dict[str, str]]] = None, entrypoint: str = "python") -> Dict[str, Any]:
    """Synchronous run: condense a follow-up into a standalone question, check the cache, run the graph."""
    if graph is None:
        from halo.agents.graph import graph as default_graph

        graph = default_graph
    cache = cache or SemanticCache.get_instance()

    standalone = condense_question(question, history)
    hit = cached_answer(question, cache, standalone)
    if hit:
        return hit

    state = graph.invoke(initial_state(standalone), config=run_config(entrypoint))
    result = summarize(question, state, standalone)
    if should_cache(result):
        cache.store(standalone, cache_payload(state))
    return result
