from src.agents.state import AgentState
from src.tools.search import get_web_search_tool

def web_search(state: AgentState) -> dict:
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents", []))
    
    try:
        tool = get_web_search_tool()
        
        if hasattr(tool, "invoke"):
            docs = tool.invoke({"query": question})
        else:
            docs = tool.run(question)
            
        # Standardize result format
        if isinstance(docs, list):
            web_results = "\n".join([d["content"] if isinstance(d, dict) and "content" in d else str(d) for d in docs])
        elif isinstance(docs, str):
            web_results = docs
        else:
            web_results = str(docs)
            
        if web_results.strip():
            documents.append(web_results)
    except Exception as e:
        print(f"Web search warning: {e}")
        documents.append(f"Web search could not retrieve additional facts: {e}")
    
    return {"documents": documents}
