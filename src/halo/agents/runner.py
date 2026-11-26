"""Single entry point for running a question through the pipeline (used by the API, CLI and Streamlit)."""
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from halo.agents.attribution import unsupported_count
from halo.agents.conversation import condense_question
from halo.agents.state import initial_state
from halo.core import metrics, usage
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
        "sentences": [{**s, "sources": link_citations(s.get("sources", []))} for s in state.get("sentences") or []],
        "unsupported_sentences": unsupported_count(state.get("sentences") or []),
        "documents": state.get("documents", []),
        "verdict": state.get("verdict"),
        "retry_count": state.get("retry_count", 0),
        "rewrite_count": state.get("rewrite_count", 0),
        "web_search_used": used_web_search(state),
        "is_fallback": state.get("is_fallback", False),
        "security_flags": state.get("security_flags") or [],
        "hops": state.get("hops") or [],
        "usage": None,
        "cached": False,
        "cached_similarity": None,
    }


def budget_stopped(question: str, standalone: str, exc: "usage.BudgetExceeded") -> Dict[str, Any]:
    """Result returned when a question hits MAX_COST_PER_QUERY_USD: no unverified answer is shown."""
    state = {"generation": (f"Stopped early: this question reached its cost limit of ${exc.budget:.4f} "
                            f"before a verified answer was ready. Raise MAX_COST_PER_QUERY_USD or ask a narrower question."),
             "confidence": 0.0, "verdict": "budget_exceeded", "is_fallback": True}
    return summarize(question, state, standalone)


def finish(result: Dict[str, Any], tracker: "usage.UsageTracker", entrypoint: str) -> Dict[str, Any]:
    """Attach usage to a result and record request metrics."""
    result["usage"] = tracker.summary()
    outcome = ("cached" if result.get("cached") else "budget_exceeded" if result.get("verdict") == "budget_exceeded"
               else "fallback" if result.get("is_fallback") else "verified")
    metrics.inc("halo_requests_total", entrypoint=entrypoint, outcome=outcome)
    metrics.observe_latency(result["usage"]["latency_ms"] / 1000, entrypoint=entrypoint)
    if result.get("cached"):
        metrics.inc("halo_cache_hits_total")
    if outcome == "budget_exceeded":
        metrics.inc("halo_budget_exceeded_total")
    if result.get("security_flags"):
        metrics.inc("halo_injection_flags_total", len(result["security_flags"]))
    return result


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
    keys = ("generation", "confidence", "citations", "sentences", "documents", "doc_metadata")
    return {k: result[k] for k in keys if k in result}


def run_query(question: str, graph=None, cache: Optional[SemanticCache] = None,
              history: Optional[List[Dict[str, str]]] = None, entrypoint: str = "python") -> Dict[str, Any]:
    """Synchronous run: condense a follow-up into a standalone question, check the cache, run the graph."""
    if graph is None:
        from halo.agents.graph import graph as default_graph

        graph = default_graph
    cache = cache or SemanticCache.get_instance()

    with usage.track() as tracker:
        standalone = question
        try:
            standalone = condense_question(question, history)
            hit = cached_answer(question, cache, standalone)
            if hit:
                return finish(hit, tracker, entrypoint)
            state = graph.invoke(initial_state(standalone), config=run_config(entrypoint))
        except usage.BudgetExceeded as exc:
            return finish(budget_stopped(question, standalone, exc), tracker, entrypoint)
        result = summarize(question, state, standalone)
        if should_cache(result):
            cache.store(standalone, cache_payload(state))
        return finish(result, tracker, entrypoint)
