import pytest

pytest.importorskip("langchain_openai")

from halo.core import llm  # noqa: E402


def test_local_base_url_is_used(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm.settings, "OPENAI_API_KEY", "")
    llm.get_llm.cache_clear()
    llm.get_embeddings.cache_clear()
    try:
        chat = llm.get_llm(0.0)
        assert "11434" in str(chat.openai_api_base)
        emb = llm.get_embeddings()
        assert "11434" in str(emb.openai_api_base)
        assert emb.check_embedding_ctx_length is False
    finally:
        llm.get_llm.cache_clear()
        llm.get_embeddings.cache_clear()


def test_routing_and_fallback(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_MODEL", "gpt-4o")
    monkeypatch.setattr(llm.settings, "FAST_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(llm.settings, "FALLBACK_LLM_MODEL", "")
    llm._chat_model.cache_clear()
    assert llm.model_for("fast") == "gpt-4o-mini" and llm.model_for("main") == "gpt-4o"
    assert llm.get_llm(0.0, task="fast").model_name == "gpt-4o-mini"
    assert type(llm.get_llm(0.0)).__name__ == "ChatOpenAI"

    monkeypatch.setattr(llm.settings, "FALLBACK_LLM_MODEL", "gpt-4.1-mini")
    chained = llm.get_llm(0.0)
    assert [f.model_name for f in chained.fallbacks] == ["gpt-4.1-mini"]
    from halo.agents.schemas import GradeAnswer

    structured = llm.structured_llm(GradeAnswer)
    assert len(structured.fallbacks) == 1
    llm._chat_model.cache_clear()


def test_every_model_tracks_usage(monkeypatch):
    from halo.core.usage import UsageCallback

    monkeypatch.setattr(llm.settings, "FALLBACK_LLM_MODEL", "gpt-4.1-mini")
    llm._chat_model.cache_clear()
    chained = llm.get_llm(0.0)
    roles = [cb.role for m in [chained.runnable, *chained.fallbacks] for cb in m.callbacks if isinstance(cb, UsageCallback)]
    assert roles == ["primary", "fallback"]
    llm._chat_model.cache_clear()
