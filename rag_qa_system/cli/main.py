from __future__ import annotations
import sys
import os
from rag_qa_system.core.schemas import QueryRequest
from rag_qa_system.agents.orchestrator_agent import RAGSupervisor
from rag_qa_system.retrieval.chunker import TextChunker
from rag_qa_system.retrieval.document_loader import DocumentLoader


def run_cli():
    print("=" * 70)
    print("🛡️  AGENTIC RAG QA SYSTEM - MULTI-AGENT QUALITY GUARDRAILS")
    print("=" * 70)

    supervisor = RAGSupervisor()
    chunker = TextChunker()

    sample = """The Agentic RAG QA System provides enterprise question-answering with zero hallucination.
Quality Agents:
1. QueryAnalyzerAgent: Handles query expansion and multi-hop decomposition.
2. RelevanceEvaluatorAgent: Evaluates context relevance score and filters irrelevant noise.
3. FaithfulnessVerifierAgent: Ensures every statement is strictly backed by retrieved evidence.
4. AnswerQualityAgent: Grades clarity, completeness, and structure.
"""
    supervisor.retriever.index_documents(chunker.chunk_text(sample, source="rag_overview.md"))

    print("Index initialized. Enter questions (or 'exit' to quit):\n")
    while True:
        try:
            query = input("Ask a question > ").strip()
            if not query or query.lower() in ["exit", "quit"]:
                break

            res = supervisor.process_query(QueryRequest(query=query))
            print("\n" + "-" * 70)
            print("💡 ANSWER:")
            print(res.answer)
            print("-" * 70)
            print("📊 AGENT QUALITY SCORECARD:")
            qr = res.quality_report
            print(f"  • Retrieval Relevance: {qr.relevance_score * 100:.1f}% [{'PASSED' if qr.relevance_passed else 'FAILED'}]")
            print(f"  • Fact Faithfulness:   {qr.faithfulness_score * 100:.1f}% [{'ZERO HALLUCINATIONS' if not qr.hallucination_detected else 'FLAGGED'}]")
            print(f"  • Answer Quality:      {qr.answer_quality_score * 100:.1f}% [{'PASSED' if qr.quality_passed else 'FAILED'}]")
            print(f"  • Overall Score:       {qr.overall_score * 100:.1f}% | Latency: {res.latency_ms:.1f}ms")
            print("-" * 70 + "\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    run_cli()
