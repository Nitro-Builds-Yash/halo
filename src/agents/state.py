from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    documents: List[str]
    doc_metadata: List[Dict[str, Any]]
    cross_encoder_confidence: float
    web_search_needed: bool
    generation: str
    confidence: float
    citations: List[Dict[str, Any]]
    retry_count: int         # Legacy alias for generation_count
    generation_count: int    # Enforced hard limit (max 1-2 generations)
    rewrite_count: int       # Enforced hard limit (max 1-2 query rewrites)
    is_fallback: bool        # Flag indicating deterministic graceful fallback activated

# GraphState alias as specified in system design
GraphState = AgentState
