from __future__ import annotations
import uuid
import time
from typing import List, Dict, Any, Optional

try:
    from pydantic import BaseModel, Field

    class DocumentChunk(BaseModel):
        chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        source: str
        content: str
        chunk_index: int = 0
        token_count: int = 0
        metadata: Dict[str, Any] = Field(default_factory=dict)
        score: float = 0.0

        def copy(self):
            return DocumentChunk(
                chunk_id=self.chunk_id,
                source=self.source,
                content=self.content,
                chunk_index=self.chunk_index,
                token_count=self.token_count,
                metadata=dict(self.metadata),
                score=self.score,
            )

    class Citation(BaseModel):
        chunk_id: str
        source: str
        snippet: str
        relevance_score: float = 0.0

    class AgentStepTrace(BaseModel):
        agent_name: str
        action: str
        input_data: Dict[str, Any] = Field(default_factory=dict)
        output_data: Dict[str, Any] = Field(default_factory=dict)
        passed: bool = True
        score: float = 1.0
        reasoning: str = ""
        duration_ms: float = 0.0
        timestamp: float = Field(default_factory=time.time)

    class QualityReport(BaseModel):
        relevance_score: float = 0.0
        relevance_passed: bool = True
        relevance_notes: str = ""
        faithfulness_score: float = 0.0
        faithfulness_passed: bool = True
        faithfulness_notes: str = ""
        hallucination_detected: bool = False
        answer_quality_score: float = 0.0
        quality_passed: bool = True
        quality_notes: str = ""
        overall_score: float = 0.0
        retries_used: int = 0
        execution_steps: List[AgentStepTrace] = Field(default_factory=list)

    class QueryRequest(BaseModel):
        query: str
        top_k: int = 5
        enable_quality_checks: bool = True
        provider: Optional[str] = None
        stream: bool = False
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        query: str
        answer: str
        citations: List[Citation] = Field(default_factory=list)
        quality_report: QualityReport = Field(default_factory=QualityReport)
        retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
        latency_ms: float = 0.0
        model: str = ""

except Exception:
    class DocumentChunk:
        def __init__(self, chunk_id=None, source="", content="", chunk_index=0, token_count=0, metadata=None, score=0.0):
            self.chunk_id = chunk_id or str(uuid.uuid4())
            self.source = source
            self.content = content
            self.chunk_index = chunk_index
            self.token_count = token_count
            self.metadata = metadata or {}
            self.score = score

        def copy(self):
            return DocumentChunk(
                chunk_id=self.chunk_id,
                source=self.source,
                content=self.content,
                chunk_index=self.chunk_index,
                token_count=self.token_count,
                metadata=dict(self.metadata),
                score=self.score,
            )

    class Citation:
        def __init__(self, chunk_id="", source="", snippet="", relevance_score=0.0):
            self.chunk_id = chunk_id
            self.source = source
            self.snippet = snippet
            self.relevance_score = relevance_score

    class AgentStepTrace:
        def __init__(self, agent_name="", action="", input_data=None, output_data=None, passed=True, score=1.0, reasoning="", duration_ms=0.0, timestamp=None):
            self.agent_name = agent_name
            self.action = action
            self.input_data = input_data or {}
            self.output_data = output_data or {}
            self.passed = passed
            self.score = score
            self.reasoning = reasoning
            self.duration_ms = duration_ms
            self.timestamp = timestamp or time.time()

    class QualityReport:
        def __init__(self, relevance_score=0.0, relevance_passed=True, relevance_notes="", faithfulness_score=0.0, faithfulness_passed=True, faithfulness_notes="", hallucination_detected=False, answer_quality_score=0.0, quality_passed=True, quality_notes="", overall_score=0.0, retries_used=0, execution_steps=None):
            self.relevance_score = relevance_score
            self.relevance_passed = relevance_passed
            self.relevance_notes = relevance_notes
            self.faithfulness_score = faithfulness_score
            self.faithfulness_passed = faithfulness_passed
            self.faithfulness_notes = faithfulness_notes
            self.hallucination_detected = hallucination_detected
            self.answer_quality_score = answer_quality_score
            self.quality_passed = quality_passed
            self.quality_notes = quality_notes
            self.overall_score = overall_score
            self.retries_used = retries_used
            self.execution_steps = execution_steps or []

    class QueryRequest:
        def __init__(self, query="", top_k=5, enable_quality_checks=True, provider=None, stream=False, session_id=None):
            self.query = query
            self.top_k = top_k
            self.enable_quality_checks = enable_quality_checks
            self.provider = provider
            self.stream = stream
            self.session_id = session_id

    class QueryResponse:
        def __init__(self, query="", answer="", citations=None, quality_report=None, retrieved_chunks=None, latency_ms=0.0, model=""):
            self.query = query
            self.answer = answer
            self.citations = citations or []
            self.quality_report = quality_report or QualityReport()
            self.retrieved_chunks = retrieved_chunks or []
            self.latency_ms = latency_ms
            self.model = model
