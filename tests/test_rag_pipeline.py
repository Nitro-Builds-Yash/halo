import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.agents.state import AgentState
from src.agents.graph import decide_to_generate, increment_retry
from src.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_decide_to_generate_flow():
    state: AgentState = {
        "question": "What is CRAG?",
        "documents": [],
        "web_search_needed": True,
        "generation": "",
        "retry_count": 0
    }
    assert decide_to_generate(state) == "transform_query"
    
    state["web_search_needed"] = False
    assert decide_to_generate(state) == "generate"

def test_increment_retry():
    state: AgentState = {
        "question": "test",
        "documents": [],
        "web_search_needed": False,
        "generation": "",
        "retry_count": 1
    }
    updated = increment_retry(state)
    assert updated["retry_count"] == 2

@patch("src.api.server.ingest_file_list")
def test_upload_documents_endpoint(mock_ingest):
    mock_ingest.return_value = (1, 3)
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("test.txt", b"Sample content for Halo agent testing.", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_documents"] == 1
    assert data["total_chunks"] == 3
