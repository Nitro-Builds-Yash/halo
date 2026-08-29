from langgraph.graph import END, StateGraph
from src.agents.state import AgentState
from src.agents.nodes.retrieval import retrieve
from src.agents.nodes.grading import grade_documents
from src.agents.nodes.query_transform import transform_query
from src.agents.nodes.web_search import web_search
from src.agents.nodes.generation import generate
from src.agents.nodes.reflection import check_hallucination_and_answer

def decide_to_generate(state: AgentState):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]
    
    if web_search_needed:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def increment_retry(state: AgentState):
    return {"retry_count": state.get("retry_count", 0) + 1}

def build_graph():
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("increment_retry", increment_retry)

    # Build graph
    workflow.set_entry_point("retrieve")
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
            "max_retries": END
        }
    )
    workflow.add_edge("increment_retry", "transform_query")
    
    return workflow.compile()

graph = build_graph()
