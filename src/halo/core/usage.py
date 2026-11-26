"""Token, cost and model-call accounting for one question.

Every chat model created by `halo.core.llm` carries a `UsageCallback`. The callback writes
into whichever `UsageTracker` is active in the current context (`with track(): ...`), so a
question's usage is collected across every node, retry and fallback, including calls that
run in worker threads (LangChain copies context into them).

Prices are estimates in US dollars per 1M tokens; override or extend them with the
MODEL_PRICES setting, e.g. MODEL_PRICES='{"my-model": [0.2, 0.8]}'.
"""
import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from halo.core import metrics
from halo.core.config import settings
from halo.core.log import get_logger

logger = get_logger(__name__)

# USD per 1M tokens (input, output). Approximate list prices; check your provider's pricing.
DEFAULT_PRICES: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
}


class BudgetExceeded(BaseException):
    """Raised before an LLM call once a question has spent its budget.

    A BaseException on purpose: the pipeline's `except Exception` handlers (grader
    fallbacks, model fallbacks) must not swallow it and keep spending.
    """

    def __init__(self, spent: float, budget: float):
        super().__init__(f"per-question budget of ${budget:.4f} reached (spent ${spent:.4f})")
        self.spent, self.budget = spent, budget


def prices() -> Dict[str, tuple]:
    table = dict(DEFAULT_PRICES)
    if settings.MODEL_PRICES:
        try:
            table.update({k: tuple(v) for k, v in json.loads(settings.MODEL_PRICES).items()})
        except Exception:
            logger.warning("MODEL_PRICES is not valid JSON; using built-in prices")
    return table


def price_for(model: str) -> Optional[tuple]:
    """Price of `model`, matching dated variants (gpt-4o-mini-2024-07-18) by longest prefix."""
    model = (model or "").lower()
    table = prices()
    if model in table:
        return table[model]
    matches = [k for k in table if model.startswith(k)]
    return table[max(matches, key=len)] if matches else None


@dataclass
class UsageTracker:
    budget_usd: float = 0.0
    started: float = field(default_factory=time.perf_counter)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fallbacks: int = 0

    def record(self, model: str, inp: int, out: int) -> None:
        price = price_for(model)
        cost = (inp * price[0] + out * price[1]) / 1_000_000 if price else 0.0
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost
        if not price:
            self.unpriced_calls += 1
        m = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["input_tokens"] += inp
        m["output_tokens"] += out
        m["cost_usd"] = round(m["cost_usd"] + cost, 6)
        metrics.inc("halo_llm_calls_total", model=model)
        metrics.inc("halo_llm_tokens_total", inp, model=model, kind="input")
        metrics.inc("halo_llm_tokens_total", out, model=model, kind="output")
        metrics.inc("halo_llm_cost_usd_total", cost, model=model)

    def check_budget(self) -> None:
        if self.budget_usd and self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(self.cost_usd, self.budget_usd)

    def summary(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_is_estimate": True,
            "unpriced_calls": self.unpriced_calls,
            "fallback_calls": self.fallbacks,
            "by_model": self.by_model,
            "latency_ms": round((time.perf_counter() - self.started) * 1000),
        }


_current: contextvars.ContextVar[Optional[UsageTracker]] = contextvars.ContextVar("halo_usage", default=None)


def current() -> Optional[UsageTracker]:
    return _current.get()


@contextmanager
def track(budget_usd: Optional[float] = None) -> Iterator[UsageTracker]:
    tracker = UsageTracker(budget_usd=settings.MAX_COST_PER_QUERY_USD if budget_usd is None else budget_usd)
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def _usage_from(response) -> List[tuple]:
    """(model, input_tokens, output_tokens) for each generation in an LLMResult."""
    out = []
    llm_output = getattr(response, "llm_output", None) or {}
    for gens in getattr(response, "generations", []) or []:
        for g in gens:
            msg = getattr(g, "message", None)
            meta = getattr(msg, "usage_metadata", None) or {}
            rmeta = getattr(msg, "response_metadata", None) or {}
            model = rmeta.get("model_name") or llm_output.get("model_name") or ""
            if meta:
                out.append((model, int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))))
    if not out and llm_output.get("token_usage"):
        tu = llm_output["token_usage"]
        out.append((llm_output.get("model_name", ""), int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))))
    return out


class UsageCallback(BaseCallbackHandler):
    """Attached to every chat model; records into the active tracker and enforces the budget."""

    raise_error = True  # so BudgetExceeded stops the call instead of being logged

    def __init__(self, model_name: str, role: str = "primary"):
        self.model_name = model_name
        self.role = role

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        tracker = current()
        if tracker:
            tracker.check_budget()
            if self.role == "fallback":
                tracker.fallbacks += 1
                metrics.inc("halo_model_fallbacks_total", model=self.model_name)

    def on_llm_end(self, response, **kwargs) -> None:
        tracker = current()
        if not tracker:
            return
        for model, inp, out in _usage_from(response) or [(self.model_name, 0, 0)]:
            tracker.record(model or self.model_name, inp, out)
