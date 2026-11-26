"""Optional LangSmith tracing. Every LLM call, retrieval step and graph node then
shows up as a trace with its latency, tokens and inputs/outputs."""
import os
from typing import Any, Dict

from halo.core.config import settings
from halo.core.log import get_logger

logger = get_logger(__name__)
_done = False


def configure_tracing() -> bool:
    """Export LangSmith settings to the environment (LangChain reads them from there). Idempotent."""
    global _done
    if _done:
        return tracing_enabled()
    _done = True
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        if settings.LANGSMITH_ENDPOINT:
            os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        logger.info("LangSmith tracing enabled (project %s)", settings.LANGSMITH_PROJECT)
    elif settings.LANGSMITH_TRACING:
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; tracing disabled")
    return tracing_enabled()


def tracing_enabled() -> bool:
    return bool(settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY)


def run_config(entrypoint: str, **metadata: Any) -> Dict[str, Any]:
    """LangChain run config: names the trace and tags it by entry point (api, stream, cli, ui)."""
    return {"run_name": "halo", "tags": ["halo", entrypoint], "metadata": {"entrypoint": entrypoint, **metadata}}
