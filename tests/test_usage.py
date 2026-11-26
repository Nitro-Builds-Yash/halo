from types import SimpleNamespace

import pytest

from halo.agents import runner
from halo.core import metrics, usage


def result(model="gpt-4o-mini-2024-07-18", inp=1000, out=200):
    msg = SimpleNamespace(usage_metadata={"input_tokens": inp, "output_tokens": out},
                          response_metadata={"model_name": model})
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output={})


def test_price_lookup_and_override(monkeypatch):
    assert usage.price_for("gpt-4o-mini-2024-07-18") == usage.DEFAULT_PRICES["gpt-4o-mini"]
    assert usage.price_for("gpt-4o-2024-08-06") == usage.DEFAULT_PRICES["gpt-4o"]  # longest prefix wins
    assert usage.price_for("llama3.1") is None
    monkeypatch.setattr(usage.settings, "MODEL_PRICES", '{"llama3.1": [0, 0], "gpt-4o-mini": [1, 2]}')
    assert usage.price_for("llama3.1") == (0, 0) and usage.price_for("gpt-4o-mini") == (1, 2)
    monkeypatch.setattr(usage.settings, "MODEL_PRICES", "not json")
    assert usage.price_for("gpt-4o-mini") == usage.DEFAULT_PRICES["gpt-4o-mini"]


def test_callback_records_into_active_tracker():
    metrics.reset()
    cb = usage.UsageCallback("gpt-4o-mini")
    cb.on_llm_end(result())  # no tracker active: ignored
    with usage.track(budget_usd=0) as t:
        cb.on_chat_model_start({}, [])
        cb.on_llm_end(result())
        cb.on_llm_end(result(model="llama3.1", inp=50, out=5))
    s = t.summary()
    assert (s["llm_calls"], s["input_tokens"], s["output_tokens"]) == (2, 1050, 205)
    assert s["cost_usd"] == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1e6)
    assert s["unpriced_calls"] == 1 and set(s["by_model"]) == {"gpt-4o-mini-2024-07-18", "llama3.1"}
    assert metrics.value("halo_llm_tokens_total", model="llama3.1", kind="input") == 50
    assert usage.current() is None


def test_token_usage_fallback_format():
    r = SimpleNamespace(generations=[[SimpleNamespace(text="x")]],
                        llm_output={"model_name": "gpt-4o", "token_usage": {"prompt_tokens": 10, "completion_tokens": 4}})
    assert usage._usage_from(r) == [("gpt-4o", 10, 4)]


def test_budget_stops_calls_and_counts_fallbacks():
    cb, fb = usage.UsageCallback("gpt-4o"), usage.UsageCallback("gpt-4o-mini", role="fallback")
    with usage.track(budget_usd=0.001) as t:
        cb.on_chat_model_start({}, [])
        cb.on_llm_end(result(model="gpt-4o", inp=1000, out=0))       # $0.0025 > budget
        with pytest.raises(usage.BudgetExceeded):
            cb.on_chat_model_start({}, [])
    assert not issubclass(usage.BudgetExceeded, Exception)  # not swallowed by `except Exception`
    with usage.track(budget_usd=0) as t:
        fb.on_chat_model_start({}, [])
    assert t.fallbacks == 1


def test_run_query_reports_usage_and_handles_budget(monkeypatch):
    metrics.reset()
    monkeypatch.setattr(runner, "condense_question", lambda q, h: q)

    class Cache:
        def lookup(self, q):
            return None

        def store(self, q, p):
            pass

    class Graph:
        def invoke(self, state, config=None):
            usage.UsageCallback("gpt-4o-mini").on_llm_end(result())
            return {**state, "generation": "30 days", "confidence": 0.9, "verdict": "useful"}

    class Broke:
        def invoke(self, state, config=None):
            raise usage.BudgetExceeded(0.02, 0.01)

    ok = runner.run_query("refund?", graph=Graph(), cache=Cache(), entrypoint="test")
    assert ok["usage"]["llm_calls"] == 1 and ok["usage"]["cost_usd"] > 0
    stopped = runner.run_query("refund?", graph=Broke(), cache=Cache(), entrypoint="test")
    assert stopped["verdict"] == "budget_exceeded" and stopped["is_fallback"] is True
    assert "$0.0100" in stopped["generation"]
    assert metrics.value("halo_requests_total", entrypoint="test", outcome="verified") == 1
    assert metrics.value("halo_budget_exceeded_total") == 1


def test_metrics_render():
    metrics.reset()
    metrics.inc("halo_requests_total", entrypoint="api", outcome="verified")
    metrics.observe_latency(1.5, entrypoint="api")
    text = metrics.render()
    assert 'halo_requests_total{entrypoint="api",outcome="verified"} 1' in text
    assert 'halo_request_latency_seconds_bucket{entrypoint="api",le="2"} 1' in text
    assert 'halo_request_latency_seconds_bucket{entrypoint="api",le="1"} 0' in text
    assert 'halo_request_latency_seconds_count{entrypoint="api"} 1' in text
    assert "# TYPE halo_request_latency_seconds histogram" in text
