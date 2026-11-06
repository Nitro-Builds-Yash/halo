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
    retry_count: int      # Hard ceiling for regenerations
    rewrite_count: int    # Hard ceiling for query rewrites
    is_fallback: bool     # Flag indicating deterministic graceful fallback activated
