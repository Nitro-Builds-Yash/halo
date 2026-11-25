from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The question to be processed by the CRAG pipeline")


class QueryResponse(BaseModel):
    question: str
    generation: str
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    documents: List[str] = Field(default_factory=list)
    verdict: Optional[str] = None
    retry_count: int = 0
    rewrite_count: int = 0
    web_search_used: bool = False
    is_fallback: bool = False
    cached: bool = False
    cached_similarity: Optional[float] = None


class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    total_chunks: int
    rejected_files: List[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    pages: int


class DocumentList(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int


class DeleteResponse(BaseModel):
    status: str
    source: str
    chunks_removed: int
