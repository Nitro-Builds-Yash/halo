import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import json

from src.agents.graph import graph
from src.api.schemas import QueryRequest, QueryResponse, IngestResponse
from src.rag.ingest import ingest_file_list

app = FastAPI(
    title="Halo Agentic CRAG API",
    description="Production Corrective RAG with Hybrid BM25+Dense Search, CrossEncoder Reranking, Query Decomposition, and Confidence Scoring.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "halo-crag-api",
        "features": [
            "hybrid_bm25_dense_search",
            "cross_encoder_reranker",
            "query_decomposition",
            "structured_citations",
            "calibrated_confidence_scoring"
        ]
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def sync_query(query: QueryRequest):
    """
    Synchronous endpoint returning answer, numerical confidence, citations, and quality metadata.
    """
    try:
        inputs = {"question": query.question, "retry_count": 0}
        result = await graph.ainvoke(inputs)
        return QueryResponse(
            question=query.question,
            generation=result.get("generation", ""),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", []),
            documents=result.get("documents", []),
            retry_count=result.get("retry_count", 0),
            web_search_needed=result.get("web_search_needed", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request, query: QueryRequest):
    """
    Server-Sent Events (SSE) streaming node transitions, decomposition steps, and final generation.
    """
    async def event_generator():
        try:
            inputs = {"question": query.question, "retry_count": 0}
            async for output in graph.astream(inputs):
                for node_name, state_update in output.items():
                    # Stream intermediate node progress
                    filtered_update = {k: v for k, v in state_update.items() if k not in ["documents", "doc_metadata"]}
                    yield {
                        "event": "node_change",
                        "data": json.dumps({"node": node_name, "update": filtered_update})
                    }
                    if "generation" in state_update and node_name == "generate":
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "content": state_update["generation"],
                                "confidence": state_update.get("confidence", 0.0),
                                "citations": state_update.get("citations", [])
                            })
                        }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            
    return EventSourceResponse(event_generator())

@app.post("/api/v1/documents/upload", response_model=IngestResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload files to both ChromaDB vector store and BM25 lexical index with page and metadata preservation.
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
