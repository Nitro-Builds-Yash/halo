from src.agents.state import AgentState
from src.rag.vectorstore import get_retriever

def retrieve(state: AgentState) -> dict:
    """
    Retrieve documents from vector store.
    """
    print("---RETRIEVE---")
    question = state["question"]
    retriever = get_retriever()
    
    documents = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in documents]
    
    return {"documents": doc_texts}
