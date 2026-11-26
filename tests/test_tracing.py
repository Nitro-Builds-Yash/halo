import os

from halo.core import tracing


def test_tracing_off_by_default(monkeypatch):
    monkeypatch.setattr(tracing, "_done", False)
    monkeypatch.setattr(tracing.settings, "LANGSMITH_TRACING", False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.configure_tracing() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_tracing_exports_environment(monkeypatch):
    monkeypatch.setattr(tracing, "_done", False)
    monkeypatch.setattr(tracing.settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(tracing.settings, "LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setattr(tracing.settings, "LANGSMITH_PROJECT", "halo-test")
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    assert tracing.configure_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "halo-test"


def test_needs_api_key(monkeypatch):
    monkeypatch.setattr(tracing, "_done", False)
    monkeypatch.setattr(tracing.settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(tracing.settings, "LANGSMITH_API_KEY", "")
    assert tracing.configure_tracing() is False


def test_run_config():
    cfg = tracing.run_config("api", user="x")
    assert cfg["run_name"] == "halo" and cfg["tags"] == ["halo", "api"]
    assert cfg["metadata"] == {"entrypoint": "api", "user": "x"}
