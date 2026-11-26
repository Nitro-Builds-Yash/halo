from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="Index of the chunk in the evaluation batch")
    is_relevant: Literal["yes", "no"] = Field(description="Whether the chunk is semantically relevant to answering the question")
    relevance_reason: Optional[str] = Field(default=None, description="Brief justification for the relevance assessment")

class BatchChunkGrading(BaseModel):
    """Single-call structured output schema to grade all borderline retrieved chunks at once."""
    grades: List[ChunkGrade] = Field(description="Relevance evaluations for each chunk in the batch")

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

class AttributedSentence(BaseModel):
    text: str = Field(description="One sentence of the answer, copied exactly")
    chunks: List[int] = Field(default_factory=list,
                              description="Numbers n of the [#n] context passages that support this sentence; empty if none")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="Direct, factual response strictly derived from context")
    confidence: float = Field(ge=0.0, le=1.0, description="Numerical confidence score between 0.0 and 1.0 based on factual grounding")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations with pages and snippets")
    sentences: List[AttributedSentence] = Field(
        default_factory=list,
        description="The answer split into its sentences, in order, each with the passage numbers that support it")

class SubQueries(BaseModel):
    sub_questions: List[str] = Field(
        description="List of 2 to 4 atomic sub-questions needed to thoroughly answer a complex, multi-hop, or comparative query."
    )


class HopDecision(BaseModel):
    enough: bool = Field(description="True if the notes already contain everything needed to answer the question")
    next_query: str = Field(default="", description="If not enough: one focused search query for the missing fact")
    missing: str = Field(default="", description="What is still missing, in a few words")
