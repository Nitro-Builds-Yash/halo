import pytest

from halo.agents.graph import decide_to_generate
from halo.agents.nodes import grading
from halo.agents.nodes import web_search as web_search_node
from halo.agents.nodes.query_decompose import needs_decomposition
from halo.agents.nodes.reflection import calibrate_confidence, route_after_reflect
from halo.agents.nodes.retrieval import normalize_rerank_score
from halo.tools import search as search_tool
from halo.tools.search import _normalize


def test_rerank_normalization_is_monotonic():
    logits = [-5, -1, 0, 0.5, 0.9, 1.0, 1.1, 2, 8]
    normed = [normalize_rerank_score(x) for x in logits]
    assert normed == sorted(normed), "a higher logit must never get a lower score"
    assert all(0 <= n <= 1 for n in normed)
    assert normalize_rerank_score(None) is None
    assert normalize_rerank_score(-1000) == pytest.approx(0.0)


def _state(scores):
    docs = [f"chunk {i}" for i in range(len(scores))]
    return {"question": "q", "documents": docs, "doc_metadata": [{"source": f"s{i}"} for i in range(len(docs))],
            "doc_scores": scores}


def test_gate_is_per_chunk(monkeypatch):
    seen = {}

    def fake_grade(q, chunks, metas):
        seen["chunks"] = chunks
        return {0}  # first borderline chunk relevant

    monkeypatch.setattr(grading, "llm_batch_grade", fake_grade)
    out = grading.grade_documents(_state([0.95, 0.6, 0.1, None]))
    # 0 auto-kept, 2 discarded, 1 and 3 borderline -> only 1 judged relevant
    assert seen["chunks"] == ["chunk 1", "chunk 3"]
    assert out["documents"] == ["chunk 0", "chunk 1"]
    assert out["web_search_needed"] is False


def test_gate_skips_llm_when_all_confident(monkeypatch):
    def boom(*a):
        raise AssertionError("LLM grader should not be called")

    monkeypatch.setattr(grading, "llm_batch_grade", boom)
    out = grading.grade_documents(_state([0.9, 0.99, 0.2]))
    assert out["documents"] == ["chunk 0", "chunk 1"]


def test_grader_failure_keeps_borderline(monkeypatch):
    def fail(*a):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(grading, "llm_batch_grade", fail)
    out = grading.grade_documents(_state([0.5, 0.1]))
    assert out["documents"] == ["chunk 0"]


def test_empty_documents_need_web_search():
    assert grading.grade_documents(_state([]))["web_search_needed"] is True


@pytest.mark.parametrize("question,expected", [
    ("What is the standard warranty?", False),          # 'standard' contains 'and'
    ("How do I understand the policy?", False),
    ("Compare plan A and plan B", True),
    ("What is the difference between X and Y?", True),
])
def test_decomposition_heuristic(question, expected):
    assert needs_decomposition(question) is expected


def test_decide_to_generate():
    assert decide_to_generate({"web_search_needed": True, "rewrite_count": 0}) == "transform_query"
    assert decide_to_generate({"web_search_needed": True, "rewrite_count": 1}) == "generate"
    assert decide_to_generate({"web_search_needed": False, "rewrite_count": 0}) == "generate"


def test_route_after_reflect():
    assert route_after_reflect({"verdict": "useful", "retry_count": 5, "rewrite_count": 5}) == "useful"
    assert route_after_reflect({"verdict": "not_grounded", "retry_count": 0, "rewrite_count": 0}) == "retry"
    assert route_after_reflect({"verdict": "not_grounded", "retry_count": 0, "rewrite_count": 1}) == "fallback"
    assert route_after_reflect({"verdict": "not_useful", "retry_count": 1, "rewrite_count": 0}) == "fallback"


def test_calibrate_confidence():
    assert calibrate_confidence(0.8, []) == 0.8
    assert calibrate_confidence(0.8, [None]) == 0.8
    assert calibrate_confidence(1.0, [0.5]) == pytest.approx(0.8)
    assert calibrate_confidence(1.7, []) == 1.0


def test_search_result_normalization():
    assert _normalize({"results": [{"content": "a", "url": "u", "title": "t"}]}, "Tavily") == [
        {"content": "a", "url": "u", "title": "t"}]
    assert _normalize([{"content": "b", "url": "v"}], "Tavily")[0]["url"] == "v"
    assert _normalize("plain text", "DuckDuckGo") == [{"content": "plain text", "url": "DuckDuckGo", "title": ""}]
    assert _normalize("", "DuckDuckGo") == []


def test_web_search_replaces_previous_web_results(monkeypatch):
    monkeypatch.setattr(search_tool, "search_web", lambda q, max_results=3: [{"content": "new", "url": "https://new", "title": ""}])
    state = {
        "question": "q", "search_query": "q",
        "documents": ["local", "old web"],
        "doc_metadata": [{"source": "a.pdf"}, {"source": "https://old", "type": "web"}],
        "doc_scores": [0.9, None],
    }
    out = web_search_node.web_search(state)
    assert out["documents"] == ["local", "new"]
    assert out["doc_metadata"][1]["source"] == "https://new"
    assert out["doc_scores"] == [0.9, None]


def test_citations_to_unknown_sources_are_dropped():
    from halo.agents.nodes.generation import filter_citations

    metas = [{"source": "policy.pdf"}, {"source": "https://example.com/a"}]
    cits = [
        {"source": "policy.pdf", "page": 2, "snippet": "30 days"},
        {"source": "Policy.PDF", "page": 2, "snippet": "30 days"},      # duplicate (case-insensitive)
        {"source": "made-up-handbook.pdf", "page": 9, "snippet": "x"},  # invented
        {"source": "https://example.com/a", "page": None, "snippet": "y"},
    ]
    kept = filter_citations(cits, metas)
    assert [c["source"] for c in kept] == ["policy.pdf", "https://example.com/a"]


def test_context_is_capped():
    from halo.agents.nodes.generation import format_context

    docs = ["a" * 5000, "b" * 5000, "c" * 5000]
    metas = [{"source": f"s{i}"} for i in range(3)]
    ctx = format_context(docs, metas, max_doc_chars=1000, max_total_chars=2500)
    assert "a" * 1000 + " [...]" in ctx and "a" * 1001 not in ctx
    assert "b" * 1000 in ctx
    assert "c" not in ctx.replace("[Source", "")  # third doc dropped by the budget
    # the first document is always included, even if it alone exceeds the budget
    assert format_context(["x" * 50], [{}], max_doc_chars=100, max_total_chars=10).count("x") == 50
