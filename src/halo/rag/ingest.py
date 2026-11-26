import argparse
import hashlib
import os
import shutil
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from halo.core.config import settings
from halo.core.log import get_logger
from halo.rag.vectorstore import get_bm25_index, get_vectorstore

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def chunk_id(source: str, page, index: int, content: str) -> str:
    raw = f"{source}|{page}|{index}|{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 byte-order mark
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # common for Windows-made text files


def load_file_with_metadata(file_path: str, source_name: str | None = None) -> List[Document]:
    """Load a .pdf / .txt / .md file into one Document per page (one for text files)."""
    ext = os.path.splitext(file_path)[1].lower()
    source = source_name or os.path.basename(file_path)

    if ext == ".pdf":
        from pypdf import PdfReader

        docs = []
        for number, page in enumerate(PdfReader(file_path).pages, start=1):  # 1-based, as readers see it
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": source, "page": number}))
        return docs
    if ext in (".txt", ".md"):
        text = _read_text(file_path)
        return [Document(page_content=text, metadata={"source": source, "page": 1})] if text.strip() else []
    raise ValueError(f"Unsupported file format: {ext}")


def list_sources():
    """Indexed documents with their chunk and page counts."""
    return get_bm25_index().sources()


def delete_source(source: str) -> int:
    """Remove all chunks of a source from both indexes. Returns the number of chunks removed."""
    vs = get_vectorstore()
    removed_dense = 0
    try:
        existing = vs.get(where={"source": source}).get("ids", [])
        if existing:
            vs.delete(ids=existing)
            removed_dense = len(existing)
    except Exception:
        logger.exception("Could not delete Chroma chunks for %s", source)
    removed_sparse = get_bm25_index().delete_source(source)
    _remove_stored_file(source)
    return max(removed_dense, removed_sparse)


def stored_file_path(source: str) -> str:
    return os.path.join(settings.documents_dir, os.path.basename(source))


def _store_file(path: str, source: str) -> None:
    """Keep the original so citations can link to it (the API serves it at /documents/{source}/file)."""
    try:
        os.makedirs(settings.documents_dir, exist_ok=True)
        shutil.copyfile(path, stored_file_path(source))
    except Exception:
        logger.exception("Could not store a copy of %s for citation links", source)


def _remove_stored_file(source: str) -> None:
    try:
        os.remove(stored_file_path(source))
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Could not remove the stored copy of %s", source)


def ingest_file_list(file_paths: List[str], source_names: List[str] | None = None) -> Tuple[int, int]:
    """Ingest files into Chroma and BM25 with shared chunk IDs.

    Re-uploading a file with the same name replaces its previous chunks instead of
    duplicating them.
    """
    source_names = source_names or [os.path.basename(p) for p in file_paths]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    total_docs, total_chunks = 0, 0
    vs = get_vectorstore()
    bm25 = get_bm25_index()

    for path, source in zip(file_paths, source_names):
        try:
            docs = load_file_with_metadata(path, source)
        except Exception:
            logger.exception("Failed to load %s", path)
            continue

        splits = splitter.split_documents(docs)
        if not splits:
            continue

        ids, seen = [], set()
        unique_splits = []
        for i, s in enumerate(splits):
            cid = chunk_id(source, s.metadata.get("page"), i, s.page_content)
            if cid in seen:
                continue
            seen.add(cid)
            s.metadata["chunk_id"] = cid
            ids.append(cid)
            unique_splits.append(s)

        delete_source(source)
        vs.add_documents(documents=unique_splits, ids=ids)
        bm25.add_documents(
            documents=[s.page_content for s in unique_splits],
            metadatas=[s.metadata for s in unique_splits],
            ids=ids,
        )
        _store_file(path, source)
        total_docs += len(docs)
        total_chunks += len(unique_splits)
        logger.info("Ingested %s: %d page(s), %d chunk(s)", source, len(docs), len(unique_splits))

    return total_docs, total_chunks


def ingest_documents(source_dir: str) -> Tuple[int, int]:
    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(source_dir)
        for f in files
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not file_paths:
        logger.warning("No supported documents found in %s (%s)", source_dir, ", ".join(SUPPORTED_EXTENSIONS))
        return 0, 0
    return ingest_file_list(file_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the hybrid Chroma + BM25 index")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents")
    args = parser.parse_args()
    n_docs, n_chunks = ingest_documents(args.dir)
    print(f"Ingested {n_docs} document page(s) into {n_chunks} chunks.")
