import pytest

from halo.agents import conversation, runner
from halo.agents.conversation import condense_question


def test_no_history_means_no_llm_call(monkeypatch):
    monkeypatch.setattr(conversation, "rewrite_standalone", lambda q, t: pytest.fail("LLM should not be called"))
    assert condense_question("What is the refund window?") == "What is the refund window?"
    assert condense_question("What is the refund window?", [{"role": "user", "content": "  "}]) == "What is the refund window?"


def test_follow_up_is_rewritten(monkeypatch):
    seen = {}

    def fake(question, transcript):
        seen["transcript"] = transcript
        return '"What is the refund window for international orders?"'

    monkeypatch.setattr(conversation, "rewrite_standalone", fake)
    history = [{"role": "user", "content": "What is the refund window?"},
               {"role": "assistant", "content": "30 days."}]
    assert condense_question("What about international orders?", history) == \
        "What is the refund window for international orders?"
    assert seen["transcript"] == "User: What is the refund window?\nAssistant: 30 days."


def test_only_recent_turns_and_truncated(monkeypatch):
    seen = {}
    monkeypatch.setattr(conversation, "rewrite_standalone", lambda q, t: seen.setdefault("t", t) or q)
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    history.append({"role": "assistant", "content": "x" * 5000})
    condense_question("and then?", history)
    lines = seen["t"].splitlines()
    assert len(lines) == conversation.MAX_TURNS
    assert len(lines[-1]) == len("Assistant: ") + conversation.MAX_TURN_CHARS


@pytest.mark.parametrize("bad", [RuntimeError("timeout"), "", "x" * 3000])
def test_failures_fall_back_to_the_message(monkeypatch, bad):
    def fake(q, t):
        if isinstance(bad, Exception):
            raise bad
        return bad

    monkeypatch.setattr(conversation, "rewrite_standalone", fake)
    assert condense_question("and that?", [{"role": "user", "content": "hi"}]) == "and that?"


class FakeCache:
    def __init__(self):
        self.looked_up, self.stored = [], []

    def lookup(self, q):
        self.looked_up.append(q)
        return None

    def store(self, q, payload):
        self.stored.append(q)


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append((state["question"], config))
        return {**state, "generation": "30 days", "confidence": 0.9, "verdict": "useful", "documents": ["d"]}


def test_run_query_uses_standalone_question_everywhere(monkeypatch):
    monkeypatch.setattr(runner, "condense_question", lambda q, h: "refund window for international orders?")
    cache, graph = FakeCache(), FakeGraph()
    result = runner.run_query("what about international?", graph=graph, cache=cache,
                              history=[{"role": "user", "content": "refund window?"}], entrypoint="test")
    assert cache.looked_up == ["refund window for international orders?"]
    assert graph.calls[0][0] == "refund window for international orders?"
    assert graph.calls[0][1]["tags"] == ["halo", "test"]
    assert cache.stored == ["refund window for international orders?"]
    assert result["question"] == "what about international?"
    assert result["standalone_question"] == "refund window for international orders?"


def test_citation_links(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.settings, "DOCUMENTS_DIR", str(tmp_path))
    (tmp_path / "policy.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_text("x")
    linked = runner.link_citations([
        {"source": "policy.pdf", "page": 4, "snippet": "a"},
        {"source": "notes.md", "page": 1, "snippet": "b"},
        {"source": "https://example.com/refunds", "page": None, "snippet": "c"},
        {"source": "deleted.pdf", "page": 2, "snippet": "d"},
        {"source": "my policy.pdf", "page": 1, "snippet": "e"},
    ])
    assert linked[0]["url"] == "/api/v1/documents/policy.pdf/file#page=4"
    assert linked[1]["url"] == "/api/v1/documents/notes.md/file"
    assert linked[2]["url"] == "https://example.com/refunds"
    assert "url" not in linked[3] and "url" not in linked[4]
