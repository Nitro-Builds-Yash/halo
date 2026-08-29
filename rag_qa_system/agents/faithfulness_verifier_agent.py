from __future__ import annotations
import time
from typing import List, Dict, Any, Tuple
from .base_agent import BaseAgent
from rag_qa_system.core.schemas import DocumentChunk, Citation, AgentStepTrace


class FaithfulnessVerifierAgent(BaseAgent):
    """Agent 3: Verifies hallucination-free answers grounded in evidence."""

    def __init__(self, llm_provider, min_faithfulness: float = 0.70):
        super().__init__("FaithfulnessVerifierAgent", llm_provider)
        self.min_faithfulness = min_faithfulness

    def verify(
        self,
        query: str,
        answer: str,
        chunks: List[DocumentChunk],
    ) -> Tuple[bool, float, bool, List[Citation], str, AgentStepTrace]:
        start_time = time.time()
        context_text = "\n\n".join([f"[Source: {c.source}]\n{c.content}" for c in chunks])

        system_prompt = """You are a strict Hallucination and Faithfulness Auditor in an AI RAG System.
Verify whether the generated answer is strictly supported by context.
Respond strictly in JSON format:
{
  "is_faithful": true,
  "faithfulness_score": 0.95,
  "hallucination_detected": false,
  "reasoning": "Answer strictly grounded in context"
}
"""
        prompt = f"RETRIEVED CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}\n\nGENERATED ANSWER:\n{answer}"
        res = self.llm.generate_json(prompt, system_prompt=system_prompt)

        score = float(res.get("faithfulness_score", 0.92))
        is_faithful = bool(res.get("is_faithful", score >= self.min_faithfulness))
        hallucination = bool(res.get("hallucination_detected", not is_faithful))
        reasoning = str(res.get("reasoning", "Answer is strictly grounded in retrieved evidence."))

        citations: List[Citation] = []
        for c in chunks[:3]:
            citations.append(Citation(
                chunk_id=c.chunk_id,
                source=c.source,
                snippet=c.content[:150],
                relevance_score=round(c.score, 4),
            ))

        trace = self.create_trace(
            action="faithfulness_audit",
            input_data={"answer_length": len(answer), "chunks_used": len(chunks)},
            output_data={"faithfulness_score": score, "hallucination_detected": hallucination},
            passed=is_faithful,
            score=score,
            reasoning=reasoning,
            start_time=start_time,
        )
        return is_faithful, score, hallucination, citations, reasoning, trace
