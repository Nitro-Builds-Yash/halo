from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional
from .base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic fallback provider for offline / test execution."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        context_match = re.search(r'CONTEXT:\s*\n(.*?)\n\nUSER QUESTION:', prompt, re.DOTALL)
        question_match = re.search(r'USER QUESTION:\s*(.*?)$', prompt, re.DOTALL)

        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else prompt

        if context:
            return f"Based on the retrieved documentation: {context[:200]}... This directly addresses: '{question}'."
        return f"Regarding '{question[:80]}', the Agentic RAG QA System guarantees high-accuracy retrieval and zero hallucination via multi-agent quality gates."

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if "relevance" in prompt.lower() or "relevance" in (system_prompt or "").lower():
            return {
                "is_relevant": True,
                "relevance_score": 0.92,
                "reasoning": "The retrieved context directly matches the query intent.",
                "relevant_indices": [0, 1]
            }
        elif "faithfulness" in prompt.lower() or "hallucination" in (system_prompt or "").lower():
            return {
                "is_faithful": True,
                "faithfulness_score": 0.95,
                "hallucination_detected": False,
                "reasoning": "All statements in the answer are strictly supported by retrieved context."
            }
        elif "quality" in prompt.lower() or "answer quality" in (system_prompt or "").lower():
            return {
                "quality_score": 0.91,
                "is_complete": True,
                "is_concise": True,
                "reasoning": "The answer is clear, well-structured, and directly answers the question."
            }
        elif "query formulation" in (system_prompt or "").lower() or "sub_queries" in prompt:
            return {
                "intent": "informational",
                "sub_queries": [prompt[:60]],
                "keywords": ["system", "features", "architecture"]
            }
        return {"status": "ok", "score": 0.90, "reasoning": "Verified"}
