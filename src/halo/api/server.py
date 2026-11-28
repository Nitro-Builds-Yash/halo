import contextvars
import json
import os
import shutil
import tempfile
import uuid
from typing import AsyncGenerator, List

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from halo import __version__ as VERSION
from halo.agents.conversation import condense_question
from halo.agents.graph import graph
from halo.agents.runner import budget_stopped, cache_payload, cached_answer, finish, should_cache, summarize
from halo.agents.state import initial_state
from halo.api.schemas import DeleteResponse, DocumentList, IngestResponse, JobResponse, QueryRequest, QueryResponse
from halo.core import metrics, usage
from halo.core.config import settings
from halo.core.log import get_logger
from halo.core.semantic_cache import SemanticCache
from halo.core.tracing import run_config, tracing_enabled
from halo.rag.bm25 import backend_name as keyword_backend
from halo.rag.extract import ocr_available
from halo.rag.ingest import SUPPORTED_EXTENSIONS, delete_source, ingest_file_list, list_sources
from halo.rag.jobs import JobRegistry

logger = get_logger(__name__)


app = FastAPI(
    title="Halo Agentic CRAG API",
    description="Corrective RAG with hybrid retrieval, cross-encoder gatekeeping, answer verification and a semantic cache.",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

semantic_cache = SemanticCache.get_instance()
ingestion_jobs = JobRegistry(ingest_file_list, on_finish=lambda: semantic_cache.clear())

# The web app (src/halo/web): index at "/", assets under /static.
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_app():
    return FileResponse(os.path.join(WEB_DIR, "index.html"), media_type="text/html")

NODE_STATUS = {
    "decompose_query": "Analyzing query structure and planning retrieval...",
    "multi_hop": "Checking whether follow-up searches are needed...",
    "retrieve": "Retrieving internal knowledge from hybrid index (Chroma + BM25)...",
    "grade_documents": "Evaluating chunk relevance (cross-encoder gate + batch grading)...",
    "transform_query": "Local context insufficient. Rewriting query for web search...",
    "web_search": "Searching the web for supplementary context...",
    "generate": "Synthesizing answer with citations...",
    "reflect": "Verifying the answer against its sources...",
    "increment_retry": "Answer not verified. Retrying with web context...",
    "graceful_fallback": "Could not verify an answer. Returning best excerpts...",
}


def require_api_key(x_api_key: str | None = Header(default=None),
                    api_key: str | None = Query(default=None, include_in_schema=False)) -> None:
    """If API_KEY is configured, every /api/v1 route requires a matching X-API-Key header
    (or `?api_key=`, so citation links can be opened directly in a browser)."""
    if settings.API_KEY and settings.API_KEY not in (x_api_key, api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def safe_filename(filename: str | None) -> str:
    """Strip any directory parts (blocks '../../x' and absolute paths) and validate the extension."""
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        raise ValueError("empty filename")
    if not name.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"unsupported file type (allowed: {', '.join(SUPPORTED_EXTENSIONS)})")
    return name


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "halo-crag-api",
        "version": VERSION,
        "auth_required": bool(settings.API_KEY),
        "tracing": tracing_enabled(),
        "keyword_backend": keyword_backend(),
        "models": {"main": settings.LLM_MODEL, "fast": settings.FAST_LLM_MODEL or settings.LLM_MODEL,
                   "fallback": settings.FALLBACK_LLM_MODEL or None},
        "injection_guard": settings.INJECTION_GUARD,
        "agentic_search": settings.AGENTIC_SEARCH,
        "ocr": ocr_available(),
        "features": [
            "semantic_cache_entrypoint",
            "hybrid_bm25_dense_rrf",
            "cross_encoder_gatekeeper",
            "single_call_batch_chunk_grading",
            "answer_verification_every_generation",
            "deterministic_fallback",
            "sse_streaming",
            "document_management",
            "multi_turn_chat",
            "citation_links",
            "sentence_level_citations",
            "web_app",
            "prompt_injection_guard",
            "usage_and_cost_tracking",
            "model_fallback_and_routing",
            "agentic_multi_hop_search",
            "table_extraction",
            "background_ingestion_jobs",
        ],
    }


def _in_context(fn, *args):
    """Run `fn` in a worker thread with the current context, so the active usage tracker
    (a ContextVar) sees the LLM calls made there."""
    ctx = contextvars.copy_context()
    return run_in_threadpool(ctx.run, fn, *args)


def message_event(result: dict) -> dict:
    return {"event": "message", "content": result["generation"], "confidence": result["confidence"],
            "citations": result["citations"], "sentences": result["sentences"],
            "unsupported_sentences": result["unsupported_sentences"], "verdict": result["verdict"],
            "fallback": result["is_fallback"], "cached": result["cached"],
            "security_flags": result["security_flags"], "hops": result["hops"], "usage": result["usage"]}


@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def sync_query(query: QueryRequest):
    with usage.track() as tracker:
        standalone = query.question
        try:
            standalone = await _in_context(condense_question, query.question, query.history_dicts())
            hit = await _in_context(cached_answer, query.question, semantic_cache, standalone)
            if hit:
                return QueryResponse(**finish(hit, tracker, "api"))
            state = await graph.ainvoke(initial_state(standalone), config=run_config("api"))
        except usage.BudgetExceeded as exc:
            return QueryResponse(**finish(budget_stopped(query.question, standalone, exc), tracker, "api"))
        except Exception:
            logger.exception("Pipeline failed")
            metrics.inc("halo_requests_total", entrypoint="api", outcome="error")
            raise HTTPException(status_code=500, detail="The question could not be processed. Check server logs.") from None

        result = summarize(query.question, state, standalone)
        if should_cache(result):
            await run_in_threadpool(semantic_cache.store, standalone, cache_payload(state))
        return QueryResponse(**finish(result, tracker, "api"))


@app.post("/api/v1/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(query: QueryRequest):
    """SSE stream of pipeline status updates. Only the final, post-verification answer is
    sent as a `message` event, so users never see an answer that is later rejected."""

    async def event_generator() -> AsyncGenerator[str, None]:
        with usage.track() as tracker:
            standalone = query.question
            state = None
            try:
                standalone = await _in_context(condense_question, query.question, query.history_dicts())
                if standalone != query.question:
                    yield sse({"event": "status", "node": "condense", "message": f"Follow-up understood as: {standalone}"})
                hit = await _in_context(cached_answer, query.question, semantic_cache, standalone)
                if hit:
                    yield sse({"event": "status", "node": "cache", "message": "Matched semantic cache",
                               "similarity": hit["cached_similarity"]})
                    yield sse(message_event(finish(hit, tracker, "stream")))
                    yield sse({"event": "done"})
                    return

                state = initial_state(standalone)
                async for event in graph.astream_events(state, version="v2", config=run_config("stream")):
                    kind, name = event.get("event"), event.get("name")
                    if name not in NODE_STATUS:
                        continue
                    if kind == "on_chain_start":
                        yield sse({"event": "status", "node": name, "message": NODE_STATUS[name]})
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            state.update(output)
                            if name == "multi_hop":
                                for hop in output.get("hops") or []:
                                    yield sse({"event": "status", "node": "hop",
                                               "message": f"Follow-up search: {hop['query']}"})
            except usage.BudgetExceeded as exc:
                yield sse(message_event(finish(budget_stopped(query.question, standalone, exc), tracker, "stream")))
                yield sse({"event": "done"})
                return
            except Exception:
                logger.exception("Streaming pipeline failed")
                metrics.inc("halo_requests_total", entrypoint="stream", outcome="error")
                yield sse({"event": "error", "error": "The question could not be processed."})
                return

            result = summarize(query.question, state, standalone)
            if should_cache(result):
                await run_in_threadpool(semantic_cache.store, standalone, cache_payload(state))
            yield sse(message_event(finish(result, tracker, "stream")))
            yield sse({"event": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus text format: requests, outcomes, latency, tokens, estimated cost, cache hits,
    injection flags, budget stops and model fallbacks."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


async def _save_uploads(files: List[UploadFile]):
    """Save valid uploads under random names in a new temp dir. Returns (dir, paths, names, rejected)."""
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    temp_dir = tempfile.mkdtemp(prefix="halo_upload_")
    saved_paths, source_names, rejected = [], [], []
    for file in files:
        try:
            name = safe_filename(file.filename)
        except ValueError as e:
            rejected.append(f"{file.filename}: {e}")
            continue

        # Saved under a random name inside temp_dir; the clean name is only used as metadata.
        path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{os.path.splitext(name)[1].lower()}")
        size = 0
        too_big = False
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    too_big = True
                    break
                out.write(chunk)
        if too_big:
            os.remove(path)
            rejected.append(f"{name}: larger than {settings.MAX_UPLOAD_MB} MB")
            continue
        saved_paths.append(path)
        source_names.append(name)
    return temp_dir, saved_paths, source_names, rejected


@app.post("/api/v1/documents/upload", dependencies=[Depends(require_api_key)],
          responses={200: {"model": IngestResponse}, 202: {"model": JobResponse}})
async def upload_documents(files: List[UploadFile] = File(...), background: bool = False):
    """Index .pdf/.txt/.md files and images (OCR). Re-uploading a filename replaces its old chunks.

    With `?background=true` the call returns 202 at once with a job; poll `/api/v1/jobs/{id}`.
    """
    temp_dir, saved_paths, source_names, rejected = await _save_uploads(files)
    if not saved_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail={"message": "No valid files uploaded", "rejected": rejected})

    if background:
        job = ingestion_jobs.submit(saved_paths, source_names, cleanup_dir=temp_dir)
        return JSONResponse(status_code=202, content={**job.to_dict(), "rejected_files": rejected})

    try:
        failed: List[str] = []
        num_docs, num_chunks = await run_in_threadpool(
            ingest_file_list, saved_paths, source_names,
            lambda event, name, info=None: failed.append(f"{name}: {info}") if event == "failed" else None)
    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Failed to ingest files. Check server logs.") from None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # New or replaced documents can change answers: drop cached ones.
    await run_in_threadpool(semantic_cache.clear)
    return IngestResponse(
        status="success",
        message=f"Ingested {num_docs} page(s) into {num_chunks} chunks.",
        total_documents=num_docs,
        total_chunks=num_chunks,
        rejected_files=rejected + failed,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    job = ingestion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No such job (jobs are kept in memory for this server process)")
    return job.to_dict()


@app.get("/api/v1/jobs", response_model=List[JobResponse], dependencies=[Depends(require_api_key)])
def list_jobs():
    return [j.to_dict() for j in ingestion_jobs.recent()]


@app.get("/api/v1/documents", response_model=DocumentList, dependencies=[Depends(require_api_key)])
async def get_documents():
    docs = await run_in_threadpool(list_sources)
    return DocumentList(documents=docs, total_chunks=sum(d["chunks"] for d in docs))


@app.delete("/api/v1/documents/{source}", response_model=DeleteResponse, dependencies=[Depends(require_api_key)])
async def remove_document(source: str):
    removed = await run_in_threadpool(delete_source, source)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No indexed document named {source!r}")
    await run_in_threadpool(semantic_cache.clear)
    return DeleteResponse(status="deleted", source=source, chunks_removed=removed)


@app.get("/api/v1/documents/{source}/file", dependencies=[Depends(require_api_key)])
async def get_document_file(source: str):
    """Serve an indexed document's original file (citation links point here, with #page=N for PDFs)."""
    name = os.path.basename(source)
    root = os.path.realpath(settings.documents_dir)
    path = os.path.realpath(os.path.join(root, name))
    if name != source or os.path.dirname(path) != root or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No stored file for {source!r}")
    media = {".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8"}.get(
        os.path.splitext(name)[1].lower(), "text/plain; charset=utf-8")
    return FileResponse(path, media_type=media, filename=name, content_disposition_type="inline")
