import pytest

from halo.agents.nodes import multi_hop
from halo.agents.schemas import HopDecision


class FakeRetriever:
    def __init__(self, corpus):
        self.corpus, self.queries = corpus, []

    def retrieve(self, query, top_k=4, use_rerank=True):
        self.queries.append(query)
        return [{"content": text, "metadata": {"source": "kb.md", "chunk_id": cid}}
                for cid, text in self.corpus.get(query, [])]


def state(**kw):
    base = {"question": "Who leads the team that built Halo and when did they join?",
            "sub_questions": ["Who built Halo?", "When did the lead join?"],
            "documents": ["Halo was built by the Search team."],
            "doc_metadata": [{"source": "kb.md", "chunk_id": "c0"}], "doc_scores": [0.9],
            "security_flags": [], "hops": []}
    return {**base, **kw}


@pytest.mark.parametrize("mode,subs,expected", [
    ("auto", ["a", "b"], True), ("auto", ["a"], False), ("on", ["a"], True), ("off", ["a", "b"], False)])
def test_should_run(monkeypatch, mode, subs, expected):
    monkeypatch.setattr(multi_hop.settings, "AGENTIC_SEARCH", mode)
    assert multi_hop.should_run({"sub_questions": subs}) is expected


def test_follow_up_searches_add_new_chunks(monkeypatch):
    monkeypatch.setattr(multi_hop.settings, "AGENTIC_SEARCH", "auto")
    monkeypatch.setattr(multi_hop.settings, "MAX_HOPS", 3)
    retr = FakeRetriever({"Search team lead": [("c1", "Priya Rao leads the Search team.")],
                          "Priya Rao start date": [("c2", "Priya Rao joined in March 2024.")]})
    monkeypatch.setattr(multi_hop, "get_hybrid_retriever", lambda: retr)
    plans = iter([HopDecision(enough=False, next_query="Search team lead", missing="team lead"),
                  HopDecision(enough=False, next_query="Priya Rao start date", missing="join date"),
                  HopDecision(enough=True)])
    monkeypatch.setattr(multi_hop, "plan_next", lambda q, notes, prev: next(plans))
    out = multi_hop.multi_hop(state())
    assert [h["query"] for h in out["hops"]] == ["Search team lead", "Priya Rao start date"]
    assert out["documents"][-1] == "Priya Rao joined in March 2024."
    assert len(out["documents"]) == len(out["doc_metadata"]) == len(out["doc_scores"]) == 3


def test_stops_on_repeat_empty_or_limit(monkeypatch):
    monkeypatch.setattr(multi_hop.settings, "AGENTIC_SEARCH", "on")
    monkeypatch.setattr(multi_hop.settings, "MAX_HOPS", 5)
    retr = FakeRetriever({})
    monkeypatch.setattr(multi_hop, "get_hybrid_retriever", lambda: retr)
    # repeated query (case-insensitive) -> stop without searching
    monkeypatch.setattr(multi_hop, "plan_next", lambda q, n, p: HopDecision(enough=False, next_query="WHO BUILT HALO?"))
    assert multi_hop.multi_hop(state())["hops"] == [] and retr.queries == []
    # a search that finds nothing new -> one hop, then stop
    monkeypatch.setattr(multi_hop, "plan_next", lambda q, n, p: HopDecision(enough=False, next_query="unknown thing"))
    assert [h["new_chunks"] for h in multi_hop.multi_hop(state())["hops"]] == [0]
    # planner failure -> keep what we have
    monkeypatch.setattr(multi_hop, "plan_next", lambda *a: (_ for _ in ()).throw(RuntimeError("timeout")))
    out = multi_hop.multi_hop(state())
    assert out["hops"] == [] and out["documents"] == ["Halo was built by the Search team."]


def test_respects_max_hops(monkeypatch):
    monkeypatch.setattr(multi_hop.settings, "AGENTIC_SEARCH", "on")
    monkeypatch.setattr(multi_hop.settings, "MAX_HOPS", 2)
    counter = iter(range(100))
    retr = FakeRetriever({})
    retr.retrieve = lambda q, **kw: [{"content": f"fact {q}", "metadata": {"chunk_id": q}}]
    monkeypatch.setattr(multi_hop, "get_hybrid_retriever", lambda: retr)
    monkeypatch.setattr(multi_hop, "plan_next", lambda *a: HopDecision(enough=False, next_query=f"q{next(counter)}"))
    assert len(multi_hop.multi_hop(state())["hops"]) == 2


def test_disabled_returns_nothing(monkeypatch):
    monkeypatch.setattr(multi_hop.settings, "AGENTIC_SEARCH", "off")
    assert multi_hop.multi_hop(state()) == {}
