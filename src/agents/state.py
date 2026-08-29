from typing import TypedDict, Annotated, List, Any
import operator

class AgentState(TypedDict):
    question: str
    documents: List[str]
    web_search_needed: bool
    generation: str
    retry_count: int
