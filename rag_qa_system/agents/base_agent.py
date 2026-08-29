from __future__ import annotations
import time
from typing import Dict, Any
from rag_qa_system.core.schemas import AgentStepTrace
from rag_qa_system.llm.base_provider import BaseLLMProvider


class BaseAgent:
    def __init__(self, name: str, llm_provider: BaseLLMProvider):
        self.name = name
        self.llm = llm_provider

    def create_trace(
        self,
        action: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        passed: bool,
        score: float,
        reasoning: str,
        start_time: float,
    ) -> AgentStepTrace:
        duration_ms = (time.time() - start_time) * 1000
        return AgentStepTrace(
            agent_name=self.name,
            action=action,
            input_data=input_data,
            output_data=output_data,
            passed=passed,
            score=round(score, 4),
            reasoning=reasoning,
            duration_ms=round(duration_ms, 2),
        )
