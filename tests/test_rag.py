import json
from types import SimpleNamespace

from halo.rag import ingest
from halo.rag.bm25 import BM25Index, content_id
from halo.rag.vectorstore import HybridRetriever

DOCS = [
    "Refund policy states returns are allowed within 30 days.",
    "Shipping takes 3 to 5 business days for domestic orders.",
]
METAS = [{"source": "policy.pdf", "page": 1}, {"source": "shipping.pdf", "page": 2}]


def test_bm25_search_and_ids(tmp_path):
    index = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    index.add_documents(DOCS, METAS, ids=["a", "b"])
    res = index.search("refund within 30 days", top_k=1)
    assert res[0]["id"] == "a"
    assert res[0]["metadata"]["chunk_id"] == "a"


def test_bm25_add_is_idempotent(tmp_path):
    index = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    index.add_documents(DOCS, METAS, ids=["a", "b"])
    index.add_documents(DOCS, METAS, ids=["a", "b"])
    assert len(index) == 2


def test_bm25_delete_source_and_persist(tmp_path):
    path = str(tmp_path / "bm25.json")
    index = BM25Index(persist_path=path)
    index.add_documents(DOCS, METAS, ids=["a", "b"])
    assert index.delete_source("policy.pdf") == 1
    reloaded = BM25Index(persist_path=path)
    assert reloaded.ids == ["b"]
    assert reloaded.search("refund") == []


def test_bm25_loads_legacy_file_without_ids(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text(json.dumps({"documents": DOCS, "metadatas": METAS}))
    index = BM25Index(persist_path=str(path))
    assert index.ids == [content_id(d) for d in DOCS]


class FakeVectorstore:
    def __init__(self, docs=None):
        self.store = {}  # id -> (text, meta)
        self.docs = docs or []

    def similarity_search_with_relevance_scores(self, query, k):
        return self.docs[:k]

    # used by ingest
    def get(self, where):
        return {"ids": [i for i, (_, m) in self.store.items() if m.get("source") == where["source"]]}

    def delete(self, ids):
        for i in ids:
            self.store.pop(i, None)

    def add_documents(self, documents, ids):
        for d, i in zip(documents, ids):
            self.store[i] = (d.page_content, d.metadata)


class NoRerank:
    def rerank(self, query, docs, top_k):
        return docs[:top_k]


def test_hybrid_fuses_on_chunk_id(tmp_path):
    bm25 = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    bm25.add_documents(DOCS, METAS, ids=["a", "b"])
    dense_hit = SimpleNamespace(page_content=DOCS[1], metadata={**METAS[1], "chunk_id": "b"})
    hr = HybridRetriever(vectorstore=FakeVectorstore([(dense_hit, 0.9)]), bm25=bm25, reranker=NoRerank())
    res = hr.retrieve("shipping business days", top_k=2)
    assert res[0]["metadata"]["chunk_id"] == "b"  # appears in both lists -> highest RRF
    assert len({r["metadata"]["chunk_id"] for r in res}) == len(res)


def test_reingest_replaces_instead_of_duplicating(tmp_path, monkeypatch):
    vs = FakeVectorstore()
    bm25 = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: vs)
    monkeypatch.setattr(ingest, "get_bm25_index", lambda: bm25)

    f = tmp_path / "upload.txt"
    f.write_text("Refunds are accepted within 30 days.\n\nShipping takes 3 to 5 days.")
    _, n1 = ingest.ingest_file_list([str(f)], ["policy.txt"])
    _, n2 = ingest.ingest_file_list([str(f)], ["policy.txt"])
    assert n1 == n2 > 0
    assert len(vs.store) == n1
    assert len(bm25) == n1

    f.write_text("Completely new text.")
    ingest.ingest_file_list([str(f)], ["policy.txt"])
    assert [t for t, _ in vs.store.values()] == ["Completely new text."]
    assert all(m["source"] == "policy.txt" for _, m in vs.store.values())


def test_bm25_sources_summary(tmp_path):
    index = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    index.add_documents(DOCS + ["More refund rules."], METAS + [{"source": "policy.pdf", "page": 3}], ids=["a", "b", "c"])
    assert index.sources() == [
        {"source": "policy.pdf", "chunks": 2, "pages": 2},
        {"source": "shipping.pdf", "chunks": 1, "pages": 1},
    ]


def test_bm25_sees_changes_from_another_process(tmp_path):
    import os
    import time

    path = str(tmp_path / "bm25.json")
    reader = BM25Index(persist_path=path)       # e.g. the Streamlit process
    writer = BM25Index(persist_path=path)       # e.g. the API process
    writer.add_documents(DOCS, METAS, ids=["a", "b"])
    assert reader.search("refund")[0]["id"] == "a"

    writer.delete_source("policy.pdf")
    os.utime(path, (time.time() + 5, time.time() + 5))  # make the mtime change unambiguous
    assert reader.search("refund") == []
    assert len(reader) == 1


def test_list_and_delete_sources(tmp_path, monkeypatch):
    vs = FakeVectorstore()
    bm25 = BM25Index(persist_path=str(tmp_path / "bm25.json"))
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: vs)
    monkeypatch.setattr(ingest, "get_bm25_index", lambda: bm25)
    f = tmp_path / "a.txt"
    f.write_text("One.\n\nTwo.")
    _, n_chunks = ingest.ingest_file_list([str(f)], ["a.txt"])
    assert n_chunks > 0
    assert ingest.list_sources() == [{"source": "a.txt", "chunks": n_chunks, "pages": 1}]
    assert ingest.delete_source("a.txt") == n_chunks
    assert ingest.list_sources() == [] and vs.store == {}
    assert ingest.delete_source("a.txt") == 0


def test_text_loading_handles_encodings(tmp_path):
    utf8 = tmp_path / "a.md"
    utf8.write_text("Café ✓", encoding="utf-8")
    win = tmp_path / "b.txt"
    win.write_bytes("Café – résumé".encode("cp1252"))
    bom16 = tmp_path / "c.txt"
    bom16.write_text("hello", encoding="utf-16")
    empty = tmp_path / "d.txt"
    empty.write_text("   ")

    assert ingest.load_file_with_metadata(str(utf8))[0].page_content == "Café ✓"
    assert ingest.load_file_with_metadata(str(win))[0].page_content == "Café – résumé"
    assert ingest.load_file_with_metadata(str(bom16))[0].page_content == "hello"
    assert ingest.load_file_with_metadata(str(empty)) == []
    doc = ingest.load_file_with_metadata(str(utf8), "renamed.md")[0]
    assert doc.metadata == {"source": "renamed.md", "page": 1}


def test_pdf_loading_uses_one_based_pages():
    import os

    import pytest

    pytest.importorskip("pypdf")
    pdf = os.path.join(os.path.dirname(__file__), "..", "docs", "interview", "HALO_CRAG_INTERVIEW_DOSSIER.pdf")
    docs = ingest.load_file_with_metadata(pdf)
    assert len(docs) > 1
    assert docs[0].metadata == {"source": "HALO_CRAG_INTERVIEW_DOSSIER.pdf", "page": 1}
    assert [d.metadata["page"] for d in docs] == sorted(d.metadata["page"] for d in docs)
