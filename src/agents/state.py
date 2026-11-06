from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    documents: List[str]
    doc_metadata: List[Dict[str, Any]]
    web_search_needed: bool
    generation: str
    confidence: float
    citations: List[Dict[str, Any]]
    retry_count: int
