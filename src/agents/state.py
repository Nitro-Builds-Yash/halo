from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str                       # the user's original question (never overwritten)
    search_query: str                   # rewritten query used for web search
    sub_questions: List[str]
    documents: List[str]
    doc_metadata: List[Dict[str, Any]]  # aligned 1:1 with documents
    doc_scores: List[Optional[float]]   # per-chunk cross-encoder score (0..1), None if unscored
    web_search_needed: bool
    generation: str
    confidence: float
    citations: List[Dict[str, Any]]
    verdict: str                        # useful | not_grounded | not_useful | no_context
    retry_count: int                    # regenerations after a failed check
    rewrite_count: int                  # query rewrites / web searches
    is_fallback: bool


GraphState = AgentState


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "search_query": question,
        "sub_questions": [],
        "documents": [],
        "doc_metadata": [],
        "doc_scores": [],
        "web_search_needed": False,
        "generation": "",
        "confidence": 0.0,
        "citations": [],
        "verdict": "",
        "retry_count": 0,
        "rewrite_count": 0,
        "is_fallback": False,
    }
