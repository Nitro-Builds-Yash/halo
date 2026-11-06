from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query or question to be processed by the CRAG pipeline")

class QueryResponse(BaseModel):
    question: str
    generation: str
    confidence: Optional[float] = Field(default=0.0, description="Numerical confidence score between 0.0 and 1.0")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted source citations with pages and snippets")
    documents: List[str] = []
    retry_count: int = 0
    rewrite_count: int = 0
    web_search_needed: bool = False
    is_fallback: bool = False
    cached: bool = False
    cached_similarity: Optional[float] = None

class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    total_chunks: int
