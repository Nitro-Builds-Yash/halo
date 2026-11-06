from typing import List, Dict, Any
from src.agents.state import AgentState
from src.rag.vectorstore import HybridRetriever
from src.core.config import settings

def retrieve(state: AgentState) -> dict:
    """
    Retrieve documents using Hybrid Search (Dense ChromaDB + BM25 Lexical + CrossEncoder Rerank)
    for each decomposed sub-query, ensuring high recall and multi-hop coverage.
    """
    print("---HYBRID RETRIEVE---")
    sub_questions = state.get("sub_questions") or [state["question"]]
    retriever = HybridRetriever()

    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    seen_contents = set()

    for q in sub_questions:
        results = retriever.retrieve(q, top_k=settings.TOP_K_RETRIEVE, use_rerank=True)
        for item in results:
            content = item.get("content", "")
            if content and content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(content)
                all_metas.append(item.get("metadata", {}))

    return {
        "documents": all_docs,
        "doc_metadata": all_metas
    }
