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

    def fake_ingest(paths, names, progress=None):
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


def test_web_app_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_metrics_endpoint_and_usage_in_response(client, monkeypatch):
    from halo.core import metrics, usage

    metrics.reset()

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            from types import SimpleNamespace

            msg = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20},
                                  response_metadata={"model_name": "gpt-4o-mini"})
            usage.UsageCallback("gpt-4o-mini").on_llm_end(
                SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output={}))
            return {**state, "generation": "30 days", "confidence": 0.9, "verdict": "useful",
                    "security_flags": [{"source": "evil.pdf", "rule": "override", "excerpt": "x", "action": "neutralized"}]}

    monkeypatch.setattr(server, "graph", FakeGraph())
    data = client.post("/api/v1/query", json={"question": "refund?"}).json()
    assert data["usage"]["input_tokens"] == 100 and data["usage"]["cost_usd"] > 0
    assert data["security_flags"][0]["source"] == "evil.pdf"
    text = client.get("/metrics").text
    assert 'halo_requests_total{entrypoint="api",outcome="verified"} 1' in text
    assert 'halo_injection_flags_total 1' in text
    assert 'halo_llm_tokens_total{kind="input",model="gpt-4o-mini"} 100' in text


def test_budget_exceeded_response(client, monkeypatch):
    from halo.core import usage

    class Broke:
        async def ainvoke(self, state, config=None):
            raise usage.BudgetExceeded(0.03, 0.02)

    monkeypatch.setattr(server, "graph", Broke())
    data = client.post("/api/v1/query", json={"question": "refund?"}).json()
    assert data["verdict"] == "budget_exceeded" and data["is_fallback"] is True


def test_health_reports_ai_settings(client):
    h = client.get("/health").json()
    assert set(h["models"]) == {"main", "fast", "fallback"}
    assert h["injection_guard"] in ("sanitize", "flag", "off")
    assert "prompt_injection_guard" in h["features"]


def test_background_upload_job(client, monkeypatch):
    import time

    from halo.rag.jobs import JobRegistry

    def slow_ingest(paths, names, progress=None):
        for n in names:
            progress("start", n)
            time.sleep(0.05)
            progress("failed" if n == "bad.txt" else "done", n, "no text found")
        return 2, 7

    cleared = []
    registry = JobRegistry(slow_ingest, on_finish=lambda: cleared.append(True))
    monkeypatch.setattr(server, "ingestion_jobs", registry)
    files = [("files", ("a.txt", b"hello", "text/plain")), ("files", ("bad.txt", b"x", "text/plain")),
             ("files", ("evil.exe", b"x", "application/octet-stream"))]
    r = client.post("/api/v1/documents/upload?background=true", files=files)
    assert r.status_code == 202
    job = r.json()
    assert job["status"] in ("queued", "running") and job["total"] == 2
    assert job["rejected_files"] and "evil.exe" in job["rejected_files"][0]

    done = registry.wait(job["id"], timeout=5)
    assert done.status == "done" and done.chunks == 7
    polled = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert polled["progress"] == 1.0 and polled["failed_files"] == ["bad.txt: no text found"]
    assert cleared, "semantic cache must be cleared after a successful job"
    assert client.get("/api/v1/jobs").json()[0]["id"] == job["id"]
    assert client.get("/api/v1/jobs/nope").status_code == 404


def test_background_job_failure_hides_details(client, monkeypatch):
    from halo.rag.jobs import JobRegistry

    def broken(paths, names, progress=None):
        raise RuntimeError("Incorrect API key provided: sk-secret")

    registry = JobRegistry(broken)
    monkeypatch.setattr(server, "ingestion_jobs", registry)
    r = client.post("/api/v1/documents/upload?background=true", files={"files": ("a.txt", b"hi", "text/plain")})
    done = registry.wait(r.json()["id"], timeout=5)
    assert done.status == "failed" and "sk-secret" not in done.error and "RuntimeError" in done.error


def test_sync_upload_reports_unreadable_files(client, monkeypatch):
    def ingest(paths, names, progress=None):
        progress("failed", names[0], "no text found")
        return 0, 0

    monkeypatch.setattr(server, "ingest_file_list", ingest)
    r = client.post("/api/v1/documents/upload", files={"files": ("scan.png", b"x", "image/png")})
    assert r.status_code == 200 and r.json()["rejected_files"] == ["scan.png: no text found"]
