from src.agents.state import AgentState
from src.core.log import get_logger
from src.tools import search as search_tool

logger = get_logger(__name__)


def web_search(state: AgentState) -> dict:
    """Add web results as separate, cited documents.

    - each result keeps its URL as `source`, so citations are meaningful
    - results from an earlier web search are replaced, not piled up, on retries
    - a failed search adds nothing (an error message is never fed to the generator as context)
    """
    query = state.get("search_query") or state["question"]

    docs, metas, scores = [], [], []
    for d, m, s in zip(state.get("documents", []),
                       state.get("doc_metadata", []),
                       state.get("doc_scores") or [None] * len(state.get("documents", []))):
        if m.get("type") != "web":
            docs.append(d)
            metas.append(m)
            scores.append(s)

    try:
        results = search_tool.search_web(query)
    except Exception:
        logger.exception("web_search: provider failed")
        results = []

    for r in results:
        docs.append(r["content"])
        metas.append({"source": r["url"], "title": r.get("title", ""), "type": "web", "page": None})
        scores.append(None)

    logger.info("web_search: %d result(s) for %r", len(results), query)
    return {"documents": docs, "doc_metadata": metas, "doc_scores": scores}
