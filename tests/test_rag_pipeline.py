import pytest
from rag_qa_system.core.schemas import QueryRequest
from rag_qa_system.retrieval.chunker import TextChunker
from rag_qa_system.retrieval.hybrid_retriever import HybridRetriever
from rag_qa_system.agents.orchestrator_agent import RAGSupervisor


def test_chunker_basic():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "Paragraph 1 is here.\n\nParagraph 2 is here with more words.\n\nParagraph 3 is final."
    chunks = chunker.chunk_text(text, source="test.txt")
    assert len(chunks) >= 1
    assert chunks[0].source == "test.txt"


def test_hybrid_retriever():
    retriever = HybridRetriever()
    chunker = TextChunker()
    doc = """Python is a high-level programming language.
FastAPI is a modern web framework for building APIs.
Docker enables containerization."""
    chunks = chunker.chunk_text(doc, source="tech.md")
    retriever.index_documents(chunks)

    results = retriever.search("FastAPI framework", top_k=2)
    assert len(results) > 0
    assert any("FastAPI" in c.content for c in results)


def test_supervisor_pipeline():
    supervisor = RAGSupervisor()
    chunker = TextChunker()
    doc = "The speed of light in vacuum is approximately 299,792,458 meters per second."
    supervisor.retriever.index_documents(chunker.chunk_text(doc, source="physics.md"))

    res = supervisor.process_query(QueryRequest(query="What is the speed of light?"))
    assert res.answer != ""
    assert res.quality_report.overall_score > 0.0
    assert len(res.quality_report.execution_steps) >= 3
