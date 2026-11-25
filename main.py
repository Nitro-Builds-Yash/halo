"""Compatibility entry point: `uvicorn main:app` (and hosts that look for main.py).

Prefer `halo serve` or `uvicorn halo.api.server:app`.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from halo.api.server import app  # noqa: E402

__all__ = ["app"]
