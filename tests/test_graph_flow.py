"""End-to-end routing tests for the CRAG graph with every LLM / search call mocked."""
import pytest

from src.agents import graph as graph_module
from src.agents.nodes import generation, grading, query_transform, reflection, retrieval
from src.agents.nodes.generation import NO_CONTEXT_ANSWER
from src.agents.schemas import AnswerWithCitations, Citation
from src.agents.state import initial_state
from src.tools import search as search_tool

QUESTION = "What is the refund window?"


class FakeRetriever:
    def __init__(self, items):
        self.items = items

    def retrieve(self, query, top_k=4, use_rerank=True):
        return self.items


def local_chunk(text="Refunds are accepted within 30 days.", logit=3.0):
    return {
        "content": text,
        "metadata": {"source": "policy.pdf", "page": 2, "chunk_id": "c1"},
        "rerank_score": logit,  # sigmoid(3.0) ~ 0.95 -> auto-kept by the gate
    }


@pytest.fixture
def pipeline(monkeypatch):
    """Configurable fake pipeline. Tests set the attributes, then call run()."""

    class P:
        local_items = [local_chunk()]
        web_results = [{"content": "Web: refunds within 30 days.", "url": "https://example.com/refunds", "title": "Refunds"}]
        web_raises = False
        grounding = [True]          # successive return values of check_grounding
        grounding_raises = False
        answers_question = True
        calls = {"web": 0, "generate": 0, "grounding": 0}

        def run(self):
            return graph_module.build_graph().invoke(initial_state(QUESTION))

    p = P()

    monkeypatch.setattr(retrieval, "get_hybrid_retriever", lambda: FakeRetriever(p.local_items))
    monkeypatch.setattr(grading, "llm_batch_grade", lambda q, chunks, metas: set(range(len(chunks))))
    monkeypatch.setattr(query_transform, "rewrite_query", lambda q: "refund window policy")

    def fake_search(query, max_results=3):
        p.calls["web"] += 1
        if p.web_raises:
            raise RuntimeError("search provider down")
        return p.web_results

    monkeypatch.setattr(search_tool, "search_web", fake_search)

    def fake_generate(question, context):
        p.calls["generate"] += 1
        assert question == QUESTION, "generation must use the user's original question"
        return AnswerWithCitations(
            answer="Refunds are accepted within 30 days.",
            confidence=0.9,
            citations=[Citation(source="policy.pdf", page=2, snippet="within 30 days")],
        )

    monkeypatch.setattr(generation, "generate_answer", fake_generate)

    def fake_grounding(context, gen):
        p.calls["grounding"] += 1
        if p.grounding_raises:
            raise RuntimeError("grader timeout")
        idx = min(p.calls["grounding"] - 1, len(p.grounding) - 1)
        return p.grounding[idx]

    monkeypatch.setattr(reflection, "check_grounding", fake_grounding)
    monkeypatch.setattr(reflection, "check_answers_question", lambda q, g: p.answers_question)
    return p


def test_local_answer_verified(pipeline):
    state = pipeline.run()
    assert state["verdict"] == "useful"
    assert state["is_fallback"] is False
    assert state["rewrite_count"] == 0
    assert pipeline.calls["web"] == 0
    # confidence blends self-rating (0.9) with the best retrieval score (~0.95)
    assert 0.9 <= state["confidence"] <= 0.95


def test_web_search_answer_is_verified_not_discarded(pipeline):
    """Regression: answers produced after a web search used to go straight to fallback."""
    pipeline.local_items = []
    state = pipeline.run()
    assert pipeline.calls["web"] == 1
    assert pipeline.calls["grounding"] == 1, "the web-based answer must be graded"
    assert state["verdict"] == "useful"
    assert state["is_fallback"] is False
    assert state["doc_metadata"][0]["source"] == "https://example.com/refunds"


def test_retry_answer_is_graded(pipeline):
    """Regression: the second answer after a retry used to skip grading."""
    pipeline.grounding = [False, True]
    state = pipeline.run()
    assert pipeline.calls["generate"] == 2
    assert pipeline.calls["grounding"] == 2
    assert state["retry_count"] == 1 and state["rewrite_count"] == 1
    assert state["verdict"] == "useful"
    assert state["is_fallback"] is False


def test_fallback_after_ceilings(pipeline):
    pipeline.grounding = [False]
    state = pipeline.run()
    assert pipeline.calls["generate"] == 2  # original + one retry, then stop
    assert state["is_fallback"] is True
    assert state["citations"], "fallback should cite the best excerpts"


def test_grader_errors_fail_closed(pipeline):
    pipeline.grounding_raises = True
    state = pipeline.run()
    assert state["is_fallback"] is True
    assert state["verdict"] == "not_grounded"


def test_question_not_overwritten_by_rewrite(pipeline):
    pipeline.local_items = []
    state = pipeline.run()
    assert state["question"] == QUESTION
    assert state["search_query"] == "refund window policy"


def test_web_search_error_is_not_used_as_context(pipeline):
    pipeline.local_items = []
    pipeline.web_raises = True
    state = pipeline.run()
    assert state["documents"] == []
    assert not any("provider down" in d for d in state["documents"])
    assert state["generation"] == NO_CONTEXT_ANSWER
    assert state["is_fallback"] is True
    assert pipeline.calls["generate"] == 0  # no LLM call without context


def test_documents_metadata_scores_stay_aligned(pipeline):
    pipeline.grounding = [False]
    state = pipeline.run()
    assert len(state["documents"]) == len(state["doc_metadata"]) == len(state["doc_scores"])
