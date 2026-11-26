"""Tiny in-process Prometheus metrics registry (text exposition format, no dependency).

Exposed at GET /metrics. Counters and a latency histogram are kept per process.
"""
import threading
from typing import Dict, Tuple

_LOCK = threading.Lock()
_COUNTERS: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120)
_HIST: Dict[Tuple[Tuple[str, str], ...], list] = {}  # labels -> [bucket counts..., sum, count]

HELP = {
    "halo_requests_total": "Questions answered, by entry point and outcome.",
    "halo_llm_calls_total": "LLM calls, by model.",
    "halo_llm_tokens_total": "LLM tokens, by model and kind (input/output).",
    "halo_llm_cost_usd_total": "Estimated LLM spend in US dollars, by model.",
    "halo_cache_hits_total": "Answers served from the semantic cache.",
    "halo_injection_flags_total": "Passages where possible prompt injection was detected.",
    "halo_budget_exceeded_total": "Questions stopped by the per-question cost budget.",
    "halo_model_fallbacks_total": "Calls answered by the fallback model after the primary failed.",
    "halo_request_latency_seconds": "End-to-end answer latency.",
}


def _key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    with _LOCK:
        k = (name, _key(labels))
        _COUNTERS[k] = _COUNTERS.get(k, 0.0) + value


def observe_latency(seconds: float, **labels: str) -> None:
    with _LOCK:
        h = _HIST.setdefault(_key(labels), [0] * len(_BUCKETS) + [0.0, 0])
        for i, b in enumerate(_BUCKETS):
            if seconds <= b:
                h[i] += 1
        h[-2] += seconds
        h[-1] += 1


def value(name: str, **labels: str) -> float:
    with _LOCK:
        return _COUNTERS.get((name, _key(labels)), 0.0)


def reset() -> None:
    with _LOCK:
        _COUNTERS.clear()
        _HIST.clear()


def _fmt_labels(labels: Tuple[Tuple[str, str], ...], extra: str = "") -> str:
    parts = [f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def render() -> str:
    lines = []
    with _LOCK:
        by_name: Dict[str, list] = {}
        for (name, labels), v in _COUNTERS.items():
            by_name.setdefault(name, []).append((labels, v))
        for name in sorted(by_name):
            lines.append(f"# HELP {name} {HELP.get(name, name)}")
            lines.append(f"# TYPE {name} counter")
            for labels, v in sorted(by_name[name]):
                lines.append(f"{name}{_fmt_labels(labels)} {v:g}")
        if _HIST:
            name = "halo_request_latency_seconds"
            lines.append(f"# HELP {name} {HELP[name]}")
            lines.append(f"# TYPE {name} histogram")
            for labels, h in sorted(_HIST.items()):
                for i, b in enumerate(_BUCKETS):
                    le = 'le="%g"' % b
                    lines.append(f"{name}_bucket{_fmt_labels(labels, le)} {h[i]}")
                inf = 'le="+Inf"'
                lines.append(f"{name}_bucket{_fmt_labels(labels, inf)} {h[-1]}")
                lines.append(f"{name}_sum{_fmt_labels(labels)} {h[-2]:g}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {h[-1]}")
    return "\n".join(lines) + "\n"
