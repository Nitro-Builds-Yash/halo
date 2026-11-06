from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on generation."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class Citation(BaseModel):
    source: str = Field(description="Document filename or external origin")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    snippet: str = Field(description="Exact excerpt verifying the answer")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="Direct, factual response strictly derived from context")
    confidence: float = Field(ge=0.0, le=1.0, description="Numerical confidence score between 0.0 and 1.0 based on factual grounding")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations with pages and snippets")

class SubQueries(BaseModel):
    sub_questions: List[str] = Field(
        description="List of 2 to 4 atomic sub-questions needed to thoroughly answer a complex, multi-hop, or comparative query."
    )
