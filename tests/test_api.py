import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import server  # noqa: E402
from src.core.semantic_cache import SemanticCache  # noqa: E402


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "semantic_cache", SemanticCache(cache_path=str(tmp_path / "c.json"), enabled=False))
    return TestClient(server.app)


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "answer_verification_every_generation" in data["features"]


def test_upload_strips_path_traversal(client, monkeypatch):
    captured = {}

    def fake_ingest(paths, names):
        captured["paths"], captured["names"] = paths, names
        return 1, 3

    monkeypatch.setattr(server, "ingest_file_list", fake_ingest)
    r = client.post("/api/v1/documents/upload", files={"files": ("../../etc/evil.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    assert captured["names"] == ["evil.pdf"]
    assert ".." not in captured["paths"][0]
    assert captured["paths"][0].startswith(server.tempfile.gettempdir())


def test_upload_rejects_unsupported_types(client, monkeypatch):
    monkeypatch.setattr(server, "ingest_file_list", lambda *a: (_ for _ in ()).throw(AssertionError("not called")))
    r = client.post("/api/v1/documents/upload", files={"files": ("run.sh", b"rm -rf /", "text/plain")})
    assert r.status_code == 400


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(server.settings, "MAX_UPLOAD_MB", 0)
    monkeypatch.setattr(server, "ingest_file_list", lambda *a: (1, 1))
    r = client.post("/api/v1/documents/upload", files={"files": ("a.txt", b"x" * 10, "text/plain")})
    assert r.status_code == 400


def test_upload_clears_cache(client, monkeypatch):
    cleared = []
    monkeypatch.setattr(server, "ingest_file_list", lambda *a: (1, 1))
    monkeypatch.setattr(server.semantic_cache, "clear", lambda: cleared.append(True))
    client.post("/api/v1/documents/upload", files={"files": ("a.txt", b"hello", "text/plain")})
    assert cleared


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(server.settings, "API_KEY", "secret")
    monkeypatch.setattr(server, "ingest_file_list", lambda *a: (1, 1))
    files = {"files": ("a.txt", b"hello", "text/plain")}
    assert client.post("/api/v1/documents/upload", files=files).status_code == 401
    ok = client.post("/api/v1/documents/upload", files=files, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_query_endpoint(client, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, state):
            return {**state, "generation": "30 days", "confidence": 0.9, "verdict": "useful",
                    "doc_metadata": [{"type": "web", "source": "https://x"}], "documents": ["d"]}

    monkeypatch.setattr(server, "graph", FakeGraph())
    data = client.post("/api/v1/query", json={"question": "refund?"}).json()
    assert data["generation"] == "30 days"
    assert data["web_search_used"] is True
    assert data["cached"] is False


def test_query_hides_internal_errors(client, monkeypatch):
    class Broken:
        async def ainvoke(self, state):
            raise RuntimeError("sk-secret-key leaked in message")

    monkeypatch.setattr(server, "graph", Broken())
    r = client.post("/api/v1/query", json={"question": "refund?"})
    assert r.status_code == 500
    assert "sk-secret" not in r.text
