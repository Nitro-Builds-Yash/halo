"""Compare the Rust BM25 backend with the pure-Python one on a synthetic corpus.

    pip install ./crates/halo-native            # build the extension (needs a Rust toolchain)
    python evals/perf/bench_native.py --docs 20000 --queries 200

Reports index build time, per-query latency and tokenization throughput for each backend.
"""
import argparse
import random
import re
import statistics
import time

_SPLIT = re.compile(r"[^\w\s]")


def make_corpus(n_docs, vocab, seed=0):
    rnd = random.Random(seed)
    words = [f"term{i}" for i in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]  # Zipf-like, as in real text
    docs = [" ".join(rnd.choices(words, weights, k=rnd.randint(80, 200))) + "." for _ in range(n_docs)]
    queries = [" ".join(rnd.choices(words, weights, k=rnd.randint(2, 6))) for _ in range(200)]
    return docs, queries


def bench(name, tokenize_many, build, search, docs, queries):
    t = time.perf_counter()
    corpus = tokenize_many(docs)
    tok_s = time.perf_counter() - t
    t = time.perf_counter()
    index = build(corpus)
    build_s = time.perf_counter() - t
    lat = []
    for q in queries:
        t = time.perf_counter()
        search(index, corpus, q)
        lat.append((time.perf_counter() - t) * 1000)
    return {"backend": name, "tokenize_s": tok_s, "build_s": build_s,
            "query_ms_p50": statistics.median(lat), "query_ms_p95": sorted(lat)[int(len(lat) * 0.95) - 1]}


def python_backend(BM25Plus):
    def tok_many(texts):
        return [_SPLIT.sub(" ", t.lower()).split() for t in texts]

    def search(index, corpus, q):
        toks = _SPLIT.sub(" ", q.lower()).split()
        scores = index.get_scores(toks)
        qs = set(toks)
        out = []
        for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            if qs.intersection(corpus[i]):
                out.append(i)
            if len(out) >= 8:
                break
        return out

    return tok_many, BM25Plus, search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    docs, queries = make_corpus(args.docs, args.vocab)
    queries = queries[: args.queries]
    rows = []

    try:
        import halo_native

        rows.append(bench(f"rust (halo-native {halo_native.__version__})", halo_native.tokenize_many,
                          halo_native.Bm25Index,
                          lambda ix, c, q: ix.top_k(halo_native.tokenize(q), 8), docs, queries))
    except ImportError:
        print("halo_native not installed: pip install ./crates/halo-native")

    try:
        from rank_bm25 import BM25Plus

        rows.append(bench("python (rank-bm25)", *python_backend(BM25Plus), docs, queries))
    except ImportError:
        print("rank-bm25 not installed: pip install rank-bm25")

    evidence = bench_evidence(docs[:2000])

    print(f"\n{args.docs:,} documents, {len(queries)} queries\n")
    print(f"{'backend':32} {'tokenize':>10} {'build':>10} {'query p50':>11} {'query p95':>11}")
    for r in rows:
        print(f"{r['backend']:32} {r['tokenize_s']:>9.2f}s {r['build_s']:>9.2f}s "
              f"{r['query_ms_p50']:>9.2f}ms {r['query_ms_p95']:>9.2f}ms")
    if len(rows) == 2:
        rust, py = rows
        print(f"\nspeed-up: build {py['build_s'] / rust['build_s']:.1f}x, "
              f"query p50 {py['query_ms_p50'] / rust['query_ms_p50']:.1f}x, "
              f"tokenize {py['tokenize_s'] / rust['tokenize_s']:.1f}x")
    if "rust" in evidence:
        print(f"\nevidence scoring (1 sentence vs 2,000 passages): rust {evidence['rust']:.2f} ms, "
              f"python {evidence['python']:.2f} ms ({evidence['python'] / evidence['rust']:.1f}x)")


def bench_evidence(passages, sentences=200):
    """Sentence-evidence scoring: one sentence against every passage, as attribution does."""
    import random as _r

    rnd = _r.Random(1)
    sents = [" ".join(rnd.choice(p.split()) for _ in range(12)) for p in rnd.sample(passages, sentences)]
    out = {}
    try:
        import halo_native

        t = time.perf_counter()
        for s in sents:
            halo_native.support_scores(s, passages)
        out["rust"] = (time.perf_counter() - t) / sentences * 1000
    except ImportError:
        pass
    pat = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?%?")

    def words(t):
        return {w for w in pat.findall(t.lower()) if len(w) > 2 or w.isdigit()}

    t = time.perf_counter()
    for s in sents:
        sw = words(s)
        [len(sw & words(p)) / len(sw) for p in passages]
    out["python"] = (time.perf_counter() - t) / sentences * 1000
    return out


if __name__ == "__main__":
    main()
