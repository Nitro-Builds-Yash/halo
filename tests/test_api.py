import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from halo.api import server  # noqa: E402
from halo.core.semantic_cache import SemanticCache  # noqa: E402


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
        async def ainvoke(self, state, config=None):
            return {**state, "generation": "30 days", "confidence": 0.9, "verdict": "useful",
                    "doc_metadata": [{"type": "web", "source": "https://x"}], "documents": ["d"]}

    monkeypatch.setattr(server, "graph", FakeGraph())
    data = client.post("/api/v1/query", json={"question": "refund?"}).json()
    assert data["generation"] == "30 days"
    assert data["web_search_used"] is True
    assert data["cached"] is False


def test_query_hides_internal_errors(client, monkeypatch):
    class Broken:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("sk-secret-key leaked in message")

    monkeypatch.setattr(server, "graph", Broken())
    r = client.post("/api/v1/query", json={"question": "refund?"})
    assert r.status_code == 500
    assert "sk-secret" not in r.text


def test_list_and_delete_documents(client, monkeypatch):
    cleared = []
    monkeypatch.setattr(server, "list_sources", lambda: [{"source": "a.pdf", "chunks": 3, "pages": 2}])
    monkeypatch.setattr(server, "delete_source", lambda s: 3 if s == "a.pdf" else 0)
    monkeypatch.setattr(server.semantic_cache, "clear", lambda: cleared.append(True))

    data = client.get("/api/v1/documents").json()
    assert data["total_chunks"] == 3 and data["documents"][0]["source"] == "a.pdf"
    assert client.delete("/api/v1/documents/a.pdf").json()["chunks_removed"] == 3
    assert cleared
    assert client.delete("/api/v1/documents/missing.pdf").status_code == 404


def test_query_with_history_returns_standalone(client, monkeypatch):
    seen = {}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            seen["question"], seen["config"] = state["question"], config
            return {**state, "generation": "45 days", "confidence": 0.9, "verdict": "useful"}

    monkeypatch.setattr(server, "graph", FakeGraph())
    monkeypatch.setattr(server, "condense_question",
                        lambda q, h: "refund window for international orders?" if h else q)
    body = {"question": "what about international?",
            "history": [{"role": "user", "content": "refund window?"}, {"role": "assistant", "content": "30 days"}]}
    data = client.post("/api/v1/query", json=body).json()
    assert data["question"] == "what about international?"
    assert data["standalone_question"] == "refund window for international orders?"
    assert seen["question"] == "refund window for international orders?"
    assert "api" in seen["config"]["tags"]
    bad = client.post("/api/v1/query", json={"question": "x", "history": [{"role": "system", "content": "y"}]})
    assert bad.status_code == 422


def test_serves_stored_files_safely(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server.settings, "DOCUMENTS_DIR", str(tmp_path / "files"))
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "policy.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "secret.txt").write_text("nope")
    r = client.get("/api/v1/documents/policy.pdf/file")
    assert r.status_code == 200 and r.content == b"%PDF-1.4 test"
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert client.get("/api/v1/documents/missing.pdf/file").status_code == 404
    assert client.get("/api/v1/documents/..%2Fsecret.txt/file").status_code == 404


def test_api_key_query_param_for_links(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server.settings, "API_KEY", "secret")
    monkeypatch.setattr(server.settings, "DOCUMENTS_DIR", str(tmp_path))
    (tmp_path / "a.txt").write_text("hi")
    assert client.get("/api/v1/documents/a.txt/file").status_code == 401
    assert client.get("/api/v1/documents/a.txt/file?api_key=secret").status_code == 200
