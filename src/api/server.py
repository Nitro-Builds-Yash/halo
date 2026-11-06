import os
import shutil
import tempfile
from typing import List, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from src.agents.graph import graph
from src.api.schemas import QueryRequest, QueryResponse, IngestResponse
from src.rag.ingest import ingest_file_list
from src.core.semantic_cache import SemanticCache

app = FastAPI(
    title="Halo Production Agentic CRAG API",
    description="Production-grade Agentic CRAG: Semantic Cache, Cross-Encoder Gatekeeper, Batch Grading, Deterministic Fallbacks, and Real-Time Event Streaming.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semantic_cache = SemanticCache.get_instance()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "halo-crag-api",
        "version": "2.0.0",
        "features": [
            "semantic_cache_entrypoint",
            "hybrid_bm25_dense_rrf",
            "cross_encoder_gatekeeper_bypass",
            "single_call_batch_chunk_grading",
            "hard_ceiling_deterministic_fallback",
            "astream_events_realtime_streaming"
        ]
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def sync_query(query: QueryRequest):
    """
    Synchronous endpoint with Semantic Cache check at entry point.
    Returns immediately if cosine similarity > 0.92 to a prior verified answer.
    """
    # 1. Semantic Cache Lookup
    cached_result = semantic_cache.lookup(query.question)
    if cached_result:
        cached_data, sim = cached_result
        return QueryResponse(
            question=query.question,
            generation=cached_data.get("generation", ""),
            confidence=cached_data.get("confidence", 1.0),
            citations=cached_data.get("citations", []),
            documents=cached_data.get("documents", []),
            retry_count=0,
            rewrite_count=0,
            web_search_needed=False,
            is_fallback=False,
            cached=True,
            cached_similarity=round(sim, 4)
        )

    # 2. Execute Full LangGraph Workflow
    try:
        inputs = {
            "question": query.question,
            "retry_count": 0,
            "rewrite_count": 0,
            "web_search_needed": False,
            "is_fallback": False
        }
        result = await graph.ainvoke(inputs)

        response_payload = {
            "generation": result.get("generation", ""),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "documents": result.get("documents", [])
        }

        # Store in cache if verified (not fallback)
        if not result.get("is_fallback", False) and result.get("confidence", 0) >= 0.70:
            semantic_cache.store(query.question, response_payload)

        return QueryResponse(
            question=query.question,
            generation=result.get("generation", ""),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", []),
            documents=result.get("documents", []),
            retry_count=result.get("retry_count", 0),
            rewrite_count=result.get("rewrite_count", 0),
            web_search_needed=result.get("web_search_needed", False),
            is_fallback=result.get("is_fallback", False),
            cached=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream")
async def chat_stream(query: QueryRequest):
    """
    StreamingResponse powered by LangGraph's astream_events(version="v2")
    Streaming user-facing real-time status transitions:
    - 'Retrieving internal knowledge...'
    - 'Local context insufficient (X/Y chunks relevant). Rewriting query for live web search...'
    - 'Verifying response against citations...'
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Check Semantic Cache first
        cached_result = semantic_cache.lookup(query.question)
        if cached_result:
            cached_data, sim = cached_result
            yield f"data: {json.dumps({'status': 'Matched semantic cache', 'similarity': round(sim, 4)})}\n\n"
            yield f"data: {json.dumps({'event': 'message', 'content': cached_data.get('generation', ''), 'confidence': cached_data.get('confidence', 1.0), 'citations': cached_data.get('citations', []), 'cached': True})}\n\n"
            return

        inputs = {
            "question": query.question,
            "retry_count": 0,
            "rewrite_count": 0,
            "web_search_needed": False,
            "is_fallback": False
        }

        status_messages = {
            "decompose_query": "Analyzing query structure and planning retrieval...",
            "retrieve": "Retrieving internal knowledge from hybrid index (Chroma + BM25)...",
            "grade_documents": "Evaluating chunk relevance via Cross-Encoder & batch grading...",
            "transform_query": "Local context insufficient. Rewriting query for live web search...",
            "web_search": "Executing fallback web search for supplementary context...",
            "generate": "Synthesizing answer with citations and confidence calibration...",
            "check_hallucination_and_answer": "Verifying response against citations & checking factual grounding...",
            "graceful_fallback": "Applying deterministic graceful fallback with best verified excerpts..."
        }

        try:
            async for event in graph.astream_events(inputs, version="v2"):
                event_type = event.get("event")
                name = event.get("name")

                # Handle node start
                if event_type == "on_chain_start" and name in status_messages:
                    msg = status_messages[name]
                    yield f"data: {json.dumps({'event': 'status', 'node': name, 'message': msg})}\n\n"

                # Handle final node outputs
                elif event_type == "on_chain_end" and name == "generate":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "generation" in output:
                        yield f"data: {json.dumps({'event': 'message', 'content': output['generation'], 'confidence': output.get('confidence', 0.8), 'citations': output.get('citations', [])})}\n\n"

                elif event_type == "on_chain_end" and name == "graceful_fallback":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "generation" in output:
                        yield f"data: {json.dumps({'event': 'message', 'content': output['generation'], 'confidence': output.get('confidence', 0.45), 'citations': output.get('citations', []), 'fallback': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/v1/documents/upload", response_model=IngestResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Ingests files simultaneously into Dense ChromaDB & Sparse BM25.
    """
    temp_dir = tempfile.mkdtemp(prefix="halo_upload_")
    saved_paths = []
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        
        num_docs, num_chunks = ingest_file_list(saved_paths)
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {num_docs} document(s) into {num_chunks} chunks with hybrid indexing.",
            total_documents=num_docs,
            total_chunks=num_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest files: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
