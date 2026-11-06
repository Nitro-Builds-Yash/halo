from typing import List, Dict, Any
from src.agents.state import AgentState
from src.rag.vectorstore import HybridRetriever
from src.core.config import settings

def retrieve(state: AgentState) -> dict:
    """
    Hybrid Retrieval + Fast Cross-Encoder Gatekeeper Pre-scoring:
    Retrieves the top candidate chunks using Dense + Sparse BM25 fused with RRF,
    and runs them through the fast local Cross-Encoder to compute confidence.
    """
    print("---HYBRID RETRIEVE + CROSS-ENCODER GATEKEEPER---")
    sub_questions = state.get("sub_questions") or [state["question"]]
    retriever = HybridRetriever()

    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    seen_contents = set()
    highest_cross_score = 0.0

    for q in sub_questions:
        results = retriever.retrieve(q, top_k=settings.TOP_K_RETRIEVE, use_rerank=True)
        for item in results:
            content = item.get("content", "")
            if content and content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(content)
                all_metas.append(item.get("metadata", {}))
                
                # Check cross-encoder rerank score
                score = item.get("rerank_score", 0.0)
                # Normalize sigmoid if logits
                norm_score = 1.0 / (1.0 + float(2.71828 ** (-score))) if score < 0 or score > 1.0 else float(score)
                if norm_score > highest_cross_score:
                    highest_cross_score = norm_score

    return {
        "documents": all_docs,
        "doc_metadata": all_metas,
        "cross_encoder_confidence": highest_cross_score
    }
