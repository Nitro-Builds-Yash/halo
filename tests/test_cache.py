import subprocess
import sys

from halo.core.semantic_cache import SemanticCache, cache_key


class FakeEmbedder:
    VECS = {"refund window?": [1.0, 0.0], "what is the refund window": [0.99, 0.05], "shipping time?": [0.0, 1.0]}

    def __init__(self):
        self.calls = 0

    def embed_query(self, q):
        self.calls += 1
        return self.VECS[q]


def make(tmp_path, **kw):
    return SemanticCache(cache_path=str(tmp_path / "c.json"), similarity_threshold=0.92,
                         embedder=FakeEmbedder(), enabled=True, **kw)


def test_hit_and_miss(tmp_path):
    cache = make(tmp_path)
    cache.store("refund window?", {"generation": "30 days"})
    hit = cache.lookup("what is the refund window")
    assert hit and hit[0]["generation"] == "30 days" and hit[1] > 0.92
    assert cache.lookup("shipping time?") is None


def test_clear_and_persistence(tmp_path):
    cache = make(tmp_path)
    cache.store("refund window?", {"generation": "30 days"})
    assert make(tmp_path).lookup("refund window?") is not None  # persisted
    cache.clear()
    assert make(tmp_path).lookup("refund window?") is None


def test_ttl_expiry(tmp_path):
    cache = make(tmp_path, ttl_seconds=-1)
    cache.store("refund window?", {"generation": "30 days"})
    assert cache.lookup("refund window?") is None


def test_query_embedded_once_per_request(tmp_path):
    cache = make(tmp_path)
    cache.store("shipping time?", {"generation": "3-5 days"})
    cache.embedder.calls = 0
    cache.lookup("refund window?")
    cache.store("refund window?", {"generation": "30 days"})
    assert cache.embedder.calls == 1


def test_disabled(tmp_path):
    cache = SemanticCache(cache_path=str(tmp_path / "c.json"), embedder=FakeEmbedder(), enabled=False)
    cache.store("refund window?", {"generation": "x"})
    assert cache.lookup("refund window?") is None


def test_cache_key_stable_across_processes():
    import os

    code = "from halo.core.semantic_cache import cache_key; print(cache_key('Refund  Window?'))"
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout.strip()
    assert out == cache_key("refund window?")


def test_clear_in_other_process_is_seen(tmp_path):
    import os
    import time

    a, b = make(tmp_path), make(tmp_path)
    a.store("refund window?", {"generation": "30 days"})
    assert b.lookup("refund window?") is not None
    a.clear()
    os.utime(a.cache_path, (time.time() + 5, time.time() + 5))
    assert b.lookup("refund window?") is None
