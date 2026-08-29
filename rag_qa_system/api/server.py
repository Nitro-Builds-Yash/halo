from __future__ import annotations
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from rag_qa_system.core.schemas import QueryRequest, QueryResponse
from rag_qa_system.agents.orchestrator_agent import RAGSupervisor
from rag_qa_system.retrieval.document_loader import DocumentLoader
from rag_qa_system.retrieval.chunker import TextChunker
from rag_qa_system.retrieval.hybrid_retriever import HybridRetriever

app = FastAPI(
    title="Agentic RAG QA System API",
    description="Production-ready RAG with Multi-Agent Quality Evaluation Guardrails",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = HybridRetriever()
supervisor = RAGSupervisor(retriever=retriever)
chunker = TextChunker()


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Agentic RAG QA System",
        "indexed_chunks": len(retriever.bm25_store.chunks),
    }


@app.post("/api/index")
async def index_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    chunks = chunker.chunk_text(text, source=file.filename)
    retriever.index_documents(chunks)
    return {
        "status": "indexed",
        "filename": file.filename,
        "chunks_created": len(chunks),
        "total_index_size": len(retriever.bm25_store.chunks),
    }


@app.post("/api/query", response_model=QueryResponse)
def query_system(req: QueryRequest):
    if not retriever.is_indexed:
        sample = """The Agentic RAG QA System is an advanced question answering platform.
It features four specialized AI quality check agents:
1. Query Formulation Agent decomposes queries.
2. Relevance Evaluator checks context precision.
3. Faithfulness Verifier detects hallucinations and extracts grounded citations.
4. Answer Quality Agent grades completeness and tone.
The system uses hybrid retrieval (BM25 + Dense Semantic Vector Search) with Cross-Encoder reranking."""
        chunks = chunker.chunk_text(sample, source="system_architecture.md")
        retriever.index_documents(chunks)

    return supervisor.process_query(req)
