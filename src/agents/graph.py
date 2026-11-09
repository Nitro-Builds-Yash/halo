from langgraph.graph import END, StateGraph
from src.agents.state import AgentState, GraphState
from src.agents.nodes.query_decompose import decompose_query
from src.agents.nodes.retrieval import retrieve
from src.agents.nodes.grading import grade_documents
from src.agents.nodes.query_transform import transform_query
from src.agents.nodes.web_search import web_search
from src.agents.nodes.generation import generate
from src.agents.nodes.reflection import check_hallucination_and_answer, MAX_REWRITES, MAX_REGENERATIONS
from src.agents.nodes.fallback import graceful_fallback

def decide_to_generate(state: GraphState):
    """
    Circuit breaker checking rewrite_count before routing to web search.
    """
    web_search_needed = state.get("web_search_needed", False)
    rewrite_count = state.get("rewrite_count", 0)

    if web_search_needed:
        if rewrite_count >= MAX_REWRITES:
            print("---CIRCUIT BREAKER: MAX REWRITES REACHED, BYPASSING WEB SEARCH---")
            return "generate"
        print("---DECISION: CONTEXT INSUFFICIENT, TRANSFORM QUERY FOR WEB SEARCH---")
        return "transform_query"
    else:
        print("---DECISION: CONTEXT SUFFICIENT, PROCEED TO GENERATE---")
        return "generate"

def increment_retry(state: GraphState):
    current_retry = state.get("retry_count", 0) + 1
    current_gen = state.get("generation_count", 0) + 1
    return {
        "retry_count": current_retry,
        "generation_count": current_gen
    }

def build_graph():
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("graceful_fallback", graceful_fallback)

    # Topological execution & Circuit Breakers
    workflow.set_entry_point("decompose_query")
    workflow.add_edge("decompose_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    
    workflow.add_conditional_edges(
        "generate",
        check_hallucination_and_answer,
        {
            "not_supported": "increment_retry",
            "not_useful": "increment_retry",
            "useful": END,
            "fallback": "graceful_fallback"
        }
    )
    workflow.add_edge("increment_retry", "transform_query")
    workflow.add_edge("graceful_fallback", END)
    
    return workflow.compile()

graph = build_graph()
