from src.agents.state import AgentState
from src.tools.search import get_web_search_tool

def web_search(state: AgentState) -> dict:
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    tool = get_web_search_tool()
    
    if hasattr(tool, "invoke"):
        docs = tool.invoke({"query": question})
    else:
        docs = tool.run(question)
        
    # Standardize result format
    web_results = "\n".join([d["content"] if isinstance(d, dict) and "content" in d else str(d) for d in docs])
    
    documents.append(web_results)
    
    return {"documents": documents}
