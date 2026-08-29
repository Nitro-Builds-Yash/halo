from __future__ import annotations
import time
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from rag_qa_system.core.schemas import AgentStepTrace


class QueryAnalyzerAgent(BaseAgent):
    """Agent 1: Decomposes query and produces search queries."""

    def __init__(self, llm_provider):
        super().__init__("QueryAnalyzerAgent", llm_provider)

    def analyze(self, query: str) -> Tuple[Dict[str, Any], AgentStepTrace]:
        start_time = time.time()
        system_prompt = """You are an expert Query Formulation Agent in an Agentic RAG system.
Decompose the query into search keywords and sub-queries.
Respond strictly in JSON format:
{
  "intent": "informational|procedural|definitional|comparative",
  "sub_queries": ["sub_query_1"],
  "keywords": ["keyword1", "keyword2"]
}
"""
        prompt = f"USER QUERY: {query}"
        res = self.llm.generate_json(prompt, system_prompt=system_prompt)

        intent = res.get("intent", "informational")
        sub_queries = res.get("sub_queries", [query])
        keywords = res.get("keywords", query.split())

        trace = self.create_trace(
            action="query_decomposition",
            input_data={"query": query},
            output_data={"intent": intent, "sub_queries": sub_queries, "keywords": keywords},
            passed=True,
            score=1.0,
            reasoning=f"Classified as {intent} with {len(sub_queries)} sub-queries.",
            start_time=start_time,
        )
        return {"intent": intent, "sub_queries": sub_queries, "keywords": keywords}, trace
