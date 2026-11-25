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
