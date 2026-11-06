import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.agents.state import AgentState
from src.agents.graph import decide_to_generate, increment_retry
from src.agents.nodes.query_decompose import decompose_query
from src.rag.bm25 import BM25Index
from src.core.semantic_cache import SemanticCache
from src.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "semantic_cache_entrypoint" in data["features"]
    assert "cross_encoder_gatekeeper_bypass" in data["features"]
    assert "single_call_batch_chunk_grading" in data["features"]

def test_semantic_cache_unit(tmp_path):
    cache_path = str(tmp_path / "test_sem_cache.json")
    cache = SemanticCache(cache_path=cache_path, similarity_threshold=0.92)
    # Test identical lookup logic
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    sim = cache._cosine_similarity(v1, v2)
    assert sim == 1.0

    v3 = [0.0, 1.0, 0.0]
    sim_ortho = cache._cosine_similarity(v1, v3)
    assert sim_ortho == 0.0

def test_bm25_index_lifecycle(tmp_path):
    index_file = str(tmp_path / "test_bm25.json")
    index = BM25Index(persist_path=index_file)
    
    docs = [
        "Refund policy states returns are allowed within 30 days.",
        "Shipping takes 3 to 5 business days for domestic orders."
    ]
    metas = [
        {"source": "policy.pdf", "page": 1},
        {"source": "shipping.pdf", "page": 2}
    ]
    index.add_documents(docs, metas)
    
    results = index.search("refund within 30 days", top_k=1)
    assert len(results) == 1
    assert "Refund policy" in results[0]["content"]

def test_decide_to_generate_with_rewrite_limits():
    # Below rewrite limit -> transforms query
    state: AgentState = {
        "question": "What is the warranty policy?",
        "sub_questions": ["What is the warranty policy?"],
        "documents": [],
        "doc_metadata": [],
        "cross_encoder_confidence": 0.3,
        "web_search_needed": True,
        "generation": "",
        "confidence": 0.0,
        "citations": [],
        "retry_count": 0,
        "rewrite_count": 0,
        "is_fallback": False
    }
    assert decide_to_generate(state) == "transform_query"

    # Reached rewrite limit -> forces generation to prevent infinite cycle
    state["rewrite_count"] = 1
    assert decide_to_generate(state) == "generate"

@patch("src.api.server.ingest_file_list")
def test_upload_documents_endpoint(mock_ingest):
    mock_ingest.return_value = (1, 4)
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("handbook.pdf", b"Content", "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
