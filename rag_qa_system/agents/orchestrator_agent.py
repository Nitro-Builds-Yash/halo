from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from rag_qa_system.core.schemas import (
    QueryRequest,
    QueryResponse,
    QualityReport,
    DocumentChunk,
    AgentStepTrace,
)
from rag_qa_system.llm.provider_factory import LLMProviderFactory
from rag_qa_system.retrieval.hybrid_retriever import HybridRetriever
from .query_analyzer_agent import QueryAnalyzerAgent
from .relevance_evaluator_agent import RelevanceEvaluatorAgent
from .faithfulness_verifier_agent import FaithfulnessVerifierAgent
from .answer_quality_agent import AnswerQualityAgent


class RAGSupervisor:
    """
    The Multi-Agent Orchestrator coordinating retrieval, synthesis, and quality check gates.
    Implements Self-Corrective RAG with automated re-querying when quality gates fail.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        provider_name: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = LLMProviderFactory.get_provider(provider_name)
        self.max_retries = max_retries

        self.query_agent = QueryAnalyzerAgent(self.llm)
        self.relevance_agent = RelevanceEvaluatorAgent(self.llm)
        self.faithfulness_agent = FaithfulnessVerifierAgent(self.llm)
        self.quality_agent = AnswerQualityAgent(self.llm)

    def process_query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        traces: List[AgentStepTrace] = []
        retries = 0
        current_query = request.query
        final_answer = ""
        citations = []
        filtered_chunks: List[DocumentChunk] = []

        rel_score = 0.0
        rel_passed = True
        rel_notes = ""

        faith_score = 0.0
        faith_passed = True
        faith_notes = ""
        hallucination = False

        qual_score = 0.0
        qual_passed = True
        qual_notes = ""

        for attempt in range(self.max_retries + 1):
            retries = attempt
            query_plan, q_trace = self.query_agent.analyze(current_query)
            traces.append(q_trace)

            retrieved = self.retriever.search(current_query, top_k=request.top_k)

            filtered_chunks, rel_score, rel_passed, rel_notes, r_trace = self.relevance_agent.evaluate(
                current_query, retrieved
            )
            traces.append(r_trace)

            if not rel_passed and attempt < self.max_retries:
                keywords = " ".join(query_plan.get("keywords", []))
                current_query = f"{request.query} {keywords}"
                continue

            context_str = "\n\n".join([f"[{c.source}]: {c.content}" for c in filtered_chunks])
            system_prompt = """You are an authoritative, accurate, and concise AI QA Assistant.
Answer the user's question strictly based on the provided context. If the answer cannot be deduced from context, state so clearly."""
            user_prompt = f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {request.query}"
            final_answer = self.llm.generate(user_prompt, system_prompt=system_prompt)

            faith_passed, faith_score, hallucination, citations, faith_notes, f_trace = self.faithfulness_agent.verify(
                request.query, final_answer, filtered_chunks
            )
            traces.append(f_trace)

            if hallucination and attempt < self.max_retries:
                continue

            qual_passed, qual_score, qual_notes, qlty_trace = self.quality_agent.grade(
                request.query, final_answer
            )
            traces.append(qlty_trace)
            break

        overall = round((rel_score * 0.3 + faith_score * 0.4 + qual_score * 0.3), 4)
        total_time = (time.time() - start_time) * 1000

        quality_report = QualityReport(
            relevance_score=rel_score,
            relevance_passed=rel_passed,
            relevance_notes=rel_notes,
            faithfulness_score=faith_score,
            faithfulness_passed=faith_passed,
            faithfulness_notes=faith_notes,
            hallucination_detected=hallucination,
            answer_quality_score=qual_score,
            quality_passed=qual_passed,
            quality_notes=qual_notes,
            overall_score=overall,
            retries_used=retries,
            execution_steps=traces,
        )

        return QueryResponse(
            query=request.query,
            answer=final_answer,
            citations=citations,
            quality_report=quality_report,
            retrieved_chunks=filtered_chunks,
            latency_ms=round(total_time, 2),
            model=getattr(self.llm, "model", "default"),
        )
