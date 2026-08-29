from __future__ import annotations
import time
from typing import List, Dict, Any, Tuple
from .base_agent import BaseAgent
from rag_qa_system.core.schemas import DocumentChunk, AgentStepTrace


class RelevanceEvaluatorAgent(BaseAgent):
    """Agent 2: Evaluates retrieved chunk relevance to filter noise."""

    def __init__(self, llm_provider, min_relevance: float = 0.60):
        super().__init__("RelevanceEvaluatorAgent", llm_provider)
        self.min_relevance = min_relevance

    def evaluate(self, query: str, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], float, bool, str, AgentStepTrace]:
        start_time = time.time()
        if not chunks:
            trace = self.create_trace(
                action="relevance_evaluation",
                input_data={"query": query, "chunk_count": 0},
                output_data={"relevant_chunks": 0},
                passed=False,
                score=0.0,
                reasoning="No context chunks retrieved.",
                start_time=start_time,
            )
            return [], 0.0, False, "No chunks retrieved", trace

        context_snippets = "\n".join([f"[{i}] {c.content[:200]}..." for i, c in enumerate(chunks)])
        system_prompt = """You are an expert Retrieval Relevance Evaluator.
Grade the relevance of the snippets to the question on a scale of 0.0 to 1.0.
Respond strictly in JSON format:
{
  "is_relevant": true,
  "relevance_score": 0.90,
  "relevant_indices": [0, 1],
  "reasoning": "Context matches query"
}
"""
        prompt = f"QUESTION: {query}\n\nRETRIEVED SNIPPETS:\n{context_snippets}"
        res = self.llm.generate_json(prompt, system_prompt=system_prompt)

        score = float(res.get("relevance_score", 0.85))
        is_relevant = bool(res.get("is_relevant", score >= self.min_relevance))
        reasoning = str(res.get("reasoning", "Evaluated retrieved context against query."))
        relevant_indices = res.get("relevant_indices", list(range(len(chunks))))

        filtered_chunks = [chunks[i] for i in relevant_indices if i < len(chunks)] or chunks

        trace = self.create_trace(
            action="relevance_evaluation",
            input_data={"query": query, "candidates": len(chunks)},
            output_data={"filtered_count": len(filtered_chunks), "relevance_score": score},
            passed=is_relevant,
            score=score,
            reasoning=reasoning,
            start_time=start_time,
        )
        return filtered_chunks, score, is_relevant, reasoning, trace
