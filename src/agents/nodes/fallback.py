from typing import List, Dict, Any
from src.agents.state import AgentState

def graceful_fallback(state: AgentState) -> dict:
    """
    Deterministic Graceful Fallback Node:
    Triggered when the pipeline reaches the hard ceiling (max_rewrites=1, max_regenerations=1)
    and the answer still cannot pass the strict factual groundedness audit.
    Returns the best verified citations along with a partial-confidence disclaimer.
    """
    print("---DETERMINISTIC GRACEFUL FALLBACK---")
    question = state["question"]
    documents = state.get("documents", [])
    doc_metadata = state.get("doc_metadata", [])

    citations: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents[:3]): # Top 3 best retrieved excerpts
        meta = doc_metadata[idx] if idx < len(doc_metadata) else {}
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        citations.append({
            "source": meta.get("source", "Indexed Document"),
            "page": meta.get("page", None),
            "snippet": snippet
        })

    fallback_text = (
        f"**Partial Confidence Disclaimer:** The system reached its self-correction limit ({state.get('retry_count', 1)} attempts) "
        f"without achieving complete factual verification for your inquiry. "
        f"Below are the most relevant grounded excerpts found in the indexed corpus. Please review the attached source citations directly."
    )

    return {
        "generation": fallback_text,
        "confidence": 0.45,
        "citations": citations,
        "is_fallback": True
    }
