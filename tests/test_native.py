"""The Rust backend must behave exactly like the pure-Python one.

Skipped unless the extension is installed (`pip install ./crates/halo-native`); CI builds it in a separate job.
"""
import math
import random
import re

import pytest

from halo.rag import bm25 as bm25_module
from halo.rag.bm25 import BM25Index

halo_native = pytest.importorskip("halo_native")

TEXTS = [
    "Refund-policy: §4.1, 30 days!", "snake_case and Café au lait", "  ...  ", "ÉCOLE Straße naïve",
    "e-mail@example.com (v2.3.0)", "数字 123 テスト", "tabs\tand\nnewlines", "", "A_B-C d'Arcy",
]


def py_tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def reference_scores(corpus, query, k1=1.5, b=0.75, delta=1.0):
    """Straight port of rank_bm25.BM25Plus.get_scores."""
    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n
    df = {}
    for d in corpus:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    out = [0.0] * n
    for q in query:
        if q not in df:
            continue
        idf = math.log((n + 1) / df[q])
        for i, d in enumerate(corpus):
            tf = d.count(q)
            out[i] += idf * (delta + tf * (k1 + 1) / (k1 * (1 - b + b * len(d) / avgdl) + tf))
    return out


def random_corpus(seed, docs=200, vocab=300):
    rnd = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    return [[rnd.choice(words) for _ in range(rnd.randint(0, 60))] for _ in range(docs)]


@pytest.mark.parametrize("text", TEXTS)
def test_tokenizer_matches_python(text):
    assert halo_native.tokenize(text) == py_tokenize(text)
    assert halo_native.tokenize_many([text, text]) == [py_tokenize(text)] * 2


@pytest.mark.parametrize("seed", range(5))
def test_scores_match_reference(seed):
    corpus = random_corpus(seed)
    query = random_corpus(seed + 100, docs=1)[0][:8] + ["missing-term"]
    native_scores = halo_native.Bm25Index(corpus).get_scores(query)
    assert native_scores == pytest.approx(reference_scores(corpus, query), rel=1e-12, abs=1e-12)


def test_scores_match_rank_bm25_when_installed():
    rank_bm25 = pytest.importorskip("rank_bm25")
    if not hasattr(rank_bm25.BM25Plus([["a"]]), "idf"):
        pytest.skip("rank_bm25 is a test stub here")
    corpus = random_corpus(7)
    query = corpus[3][:5]
    expected = list(rank_bm25.BM25Plus(corpus).get_scores(query))
    assert halo_native.Bm25Index(corpus).get_scores(query) == pytest.approx(expected, rel=1e-12)


def test_top_k_matches_python_ranking():
    corpus = random_corpus(11)
    index = halo_native.Bm25Index(corpus)
    for query in (corpus[0][:3], corpus[5][:6], ["w1", "w1", "w2"], ["nothing"]):
        scores = reference_scores(corpus, query)
        qs = set(query)
        expected = [i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    if qs.intersection(corpus[i])][:10]
        assert [i for i, _ in index.top_k(query, 10)] == expected


def test_empty_index():
    index = halo_native.Bm25Index([])
    assert len(index) == 0 and index.get_scores(["a"]) == [] and index.top_k(["a"], 3) == []


def test_index_results_identical_across_backends(tmp_path, monkeypatch):
    docs = ["Refunds are accepted within 30 days of purchase.", "Shipping takes 3 to 5 business days.",
            "International refunds take 45 days.", "Gift cards cannot be refunded.", "Section 4.1: returns"]
    metas = [{"source": f"s{i}.pdf"} for i in range(len(docs))]

    def run(native):
        monkeypatch.setattr(bm25_module, "native", native)
        index = BM25Index(persist_path=str(tmp_path / f"bm25-{bool(native)}.json"))
        index.add_documents(docs, metas, ids=[f"c{i}" for i in range(len(docs))])
        return [[(r["id"], round(r["score"], 9)) for r in index.search(q, top_k=3)]
                for q in ("refund days", "section 4.1", "shipping", "unrelated")]

    rank_bm25 = pytest.importorskip("rank_bm25")
    if not hasattr(rank_bm25.BM25Plus([["a"]]), "idf"):
        pytest.skip("rank_bm25 is a test stub here")
    assert run(halo_native) == run(None)


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HALO_NATIVE", "0")
    assert bm25_module._load_native() is None
    monkeypatch.setenv("HALO_NATIVE", "1")
    assert bm25_module._load_native() is halo_native


# ---- sentence-evidence scoring and rank fusion -------------------------------------------

EVIDENCE_TEXTS = TEXTS + [
    "Refunds are accepted within 30 days; a 2.5% fee applies to 50% of orders (v2.3.0).",
    "The quick brown fox jumps over the lazy dog 1.2.3 times", "İstanbul CAFÉ 100%% off", "ab 12 cd 3",
]


def py_content_words(text):
    from halo.agents import attribution

    return [w for w in attribution._WORD.findall(text.lower())
            if w not in attribution._STOPWORDS and (len(w) > 2 or w.isdigit())]


@pytest.mark.parametrize("text", EVIDENCE_TEXTS)
def test_content_words_match_python(text):
    assert halo_native.content_words(text) == py_content_words(text)


def test_support_scores_match_python():
    rnd = random.Random(3)
    vocab = ["refund", "days", "30", "45", "2.5%", "policy", "the", "shipping", "café", "orders", "fee"]
    for _ in range(300):
        sentence = " ".join(rnd.choices(vocab, k=rnd.randint(0, 8)))
        passages = [" ".join(rnd.choices(vocab, k=rnd.randint(0, 20))) for _ in range(5)]
        sw = set(py_content_words(sentence))
        expected = [len(sw & set(py_content_words(p))) / len(sw) if sw else 0.0 for p in passages]
        assert halo_native.support_scores(sentence, passages) == expected
        assert [halo_native.support_score(sentence, p) for p in passages] == expected


def test_rrf_matches_python(monkeypatch):
    from halo.rag import vectorstore

    rnd = random.Random(5)
    for _ in range(50):
        rankings = [rnd.sample([f"c{i}" for i in range(30)], rnd.randint(0, 12)) for _ in range(rnd.randint(0, 3))]
        monkeypatch.setattr(vectorstore, "native", None)
        expected = vectorstore.rrf_fuse(rankings, 60)
        monkeypatch.setattr(vectorstore, "native", halo_native)
        assert vectorstore.rrf_fuse(rankings, 60) == expected


def test_attribution_identical_across_backends(monkeypatch):
    from halo.agents import attribution

    docs = ["Refunds are accepted within 30 days of purchase. A receipt is required.",
            "International orders can be returned within 45 days of delivery.",
            "Gift cards cannot be exchanged for cash."]
    metas = [{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 2}, {"source": "b.md", "page": 1}]
    answer = ("Refunds are accepted within 30 days of purchase. International orders get 60 days. "
              "Gift cards cannot be exchanged for cash.")
    monkeypatch.setattr(attribution, "native", None)
    expected = attribution.attribute(answer, None, docs, metas)
    monkeypatch.setattr(attribution, "native", halo_native)
    assert attribution.attribute(answer, None, docs, metas) == expected
