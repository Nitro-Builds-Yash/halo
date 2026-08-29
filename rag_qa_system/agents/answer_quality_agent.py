from __future__ import annotations
import time
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from rag_qa_system.core.schemas import AgentStepTrace


class AnswerQualityAgent(BaseAgent):
    """Agent 4: Grades answer completeness, tone, conciseness, and clarity."""

    def __init__(self, llm_provider, min_quality: float = 0.75):
        super().__init__("AnswerQualityAgent", llm_provider)
        self.min_quality = min_quality

    def grade(self, query: str, answer: str) -> Tuple[bool, float, str, AgentStepTrace]:
        start_time = time.time()
        system_prompt = """You are an expert Answer Quality Grader.
Assess the response for completeness, conciseness, formatting, and helpfulness.
Respond strictly in JSON format:
{
  "quality_score": 0.90,
  "is_complete": true,
  "is_concise": true,
  "reasoning": "High-quality, complete answer"
}
"""
        prompt = f"USER QUERY: {query}\n\nANSWER:\n{answer}"
        res = self.llm.generate_json(prompt, system_prompt=system_prompt)

        score = float(res.get("quality_score", 0.90))
        passed = bool(score >= self.min_quality)
        reasoning = str(res.get("reasoning", "Answer meets quality and completeness standards."))

        trace = self.create_trace(
            action="quality_grading",
            input_data={"query": query, "answer_length": len(answer)},
            output_data={"quality_score": score, "passed": passed},
            passed=passed,
            score=score,
            reasoning=reasoning,
            start_time=start_time,
        )
        return passed, score, reasoning, trace
