import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.agents.state import AgentState
from src.agents.graph import decide_to_generate, increment_retry
from src.agents.nodes.query_decompose import decompose_query
from src.rag.bm25 import BM25Index
from src.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "hybrid_bm25_dense_search" in data["features"]

def test_bm25_index_lifecycle(tmp_path):
    index_file = str(tmp_path / "test_bm25.json")
    index = BM25Index(persist_path=index_file)
    
    docs = [
        "Refund policy states returns are allowed within 30 days.",
        "Shipping takes 3 to 5 business days for domestic orders.",
        "Warranty covers technical defects for twelve months."
    ]
    metas = [
        {"source": "policy.pdf", "page": 1},
        {"source": "shipping.pdf", "page": 2},
        {"source": "warranty.pdf", "page": 3}
    ]
    index.add_documents(docs, metas)
    
    # Test Search
    results = index.search("refund within 30 days", top_k=1)
    assert len(results) == 1
    assert "Refund policy" in results[0]["content"]
    assert results[0]["metadata"]["source"] == "policy.pdf"

def test_decide_to_generate():
    state: AgentState = {
        "question": "What is the warranty policy?",
        "sub_questions": ["What is the warranty policy?"],
        "documents": [],
        "doc_metadata": [],
        "web_search_needed": True,
        "generation": "",
        "confidence": 0.0,
        "citations": [],
        "retry_count": 0
    }
    assert decide_to_generate(state) == "transform_query"

    state["web_search_needed"] = False
    assert decide_to_generate(state) == "generate"

def test_increment_retry():
    state: AgentState = {
        "question": "test",
        "sub_questions": ["test"],
        "documents": [],
        "doc_metadata": [],
        "web_search_needed": False,
        "generation": "",
        "confidence": 0.0,
        "citations": [],
        "retry_count": 1
    }
    updated = increment_retry(state)
    assert updated["retry_count"] == 2

@patch("src.api.server.ingest_file_list")
def test_upload_documents_endpoint(mock_ingest):
    mock_ingest.return_value = (1, 4)
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("company_handbook.pdf", b"Dummy PDF binary contents", "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_documents"] == 1
    assert data["total_chunks"] == 4
