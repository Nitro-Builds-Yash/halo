from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import asyncio
import json

from src.agents.graph import graph
from src.api.schemas import QueryRequest, QueryResponse

app = FastAPI(title="Halo Agentic CRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request, query: QueryRequest):
    async def event_generator():
        try:
            inputs = {"question": query.question, "retry_count": 0}
            async for output in graph.astream(inputs):
                # LangGraph yields a dictionary mapping node_name -> state_updates
                for node_name, state_update in output.items():
                    yield {
                        "event": "node_change",
                        "data": json.dumps({"node": node_name})
                    }
                    if "generation" in state_update and node_name == "generate":
                        yield {
                            "event": "message",
                            "data": json.dumps({"content": state_update["generation"]})
                        }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            
    return EventSourceResponse(event_generator())
