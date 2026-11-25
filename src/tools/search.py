from typing import Dict, List

from src.core.config import settings
from src.core.log import get_logger

logger = get_logger(__name__)


def _normalize(raw, provider: str) -> List[Dict[str, str]]:
    """Turn the different result shapes of Tavily/DuckDuckGo wrappers into
    a list of {"content", "url", "title"} dicts."""
    if isinstance(raw, dict) and "results" in raw:  # langchain-tavily TavilySearch
        raw = raw["results"]
    results: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for r in raw:
            if isinstance(r, dict):
                content = r.get("content") or r.get("snippet") or ""
                if content.strip():
                    results.append({
                        "content": content.strip(),
                        "url": r.get("url") or r.get("link") or provider,
                        "title": r.get("title", ""),
                    })
            elif str(r).strip():
                results.append({"content": str(r).strip(), "url": provider, "title": ""})
    elif isinstance(raw, str) and raw.strip():
        results.append({"content": raw.strip(), "url": provider, "title": ""})
    return results


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Run a web search. Raises on provider errors; callers decide how to handle them."""
    if settings.TAVILY_API_KEY:
        try:
            from langchain_tavily import TavilySearch

            tool = TavilySearch(max_results=max_results, tavily_api_key=settings.TAVILY_API_KEY)
        except ImportError:  # older installs
            from langchain_community.tools.tavily_search import TavilySearchResults

            tool = TavilySearchResults(max_results=max_results)
        return _normalize(tool.invoke({"query": query}), "Tavily")[:max_results]

    logger.warning("TAVILY_API_KEY not set; falling back to DuckDuckGo")
    from langchain_community.tools import DuckDuckGoSearchRun

    return _normalize(DuckDuckGoSearchRun().invoke(query), "DuckDuckGo")[:max_results]
