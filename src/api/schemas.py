from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query or question to be processed by the CRAG pipeline")

class QueryResponse(BaseModel):
    question: str
    generation: str
    documents: List[str] = []
    retry_count: int = 0
    web_search_needed: bool = False

class StreamChunk(BaseModel):
    node: str
    content: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    total_chunks: int
