from langgraph.graph import END, StateGraph

from halo.agents.nodes.fallback import graceful_fallback
from halo.agents.nodes.generation import generate
from halo.agents.nodes.grading import grade_documents
from halo.agents.nodes.query_decompose import decompose_query
from halo.agents.nodes.query_transform import transform_query
from halo.agents.nodes.reflection import reflect, route_after_reflect
from halo.agents.nodes.retrieval import retrieve
from halo.agents.nodes.web_search import web_search
from halo.agents.state import GraphState
from halo.core.config import settings
from halo.core.log import get_logger

logger = get_logger(__name__)


def decide_to_generate(state: GraphState) -> str:
    """After grading: go to web search if local context is insufficient and a rewrite is still allowed."""
    if state.get("web_search_needed", False) and state.get("rewrite_count", 0) < settings.MAX_REWRITES:
        return "transform_query"
    return "generate"


def increment_retry(state: GraphState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("reflect", reflect)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("graceful_fallback", graceful_fallback)

    workflow.set_entry_point("decompose_query")
    workflow.add_edge("decompose_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"useful": END, "retry": "increment_retry", "fallback": "graceful_fallback"},
    )
    workflow.add_edge("increment_retry", "transform_query")
    workflow.add_edge("graceful_fallback", END)

    return workflow.compile()


graph = build_graph()
