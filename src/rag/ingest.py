import argparse
import hashlib
import os
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.log import get_logger
from src.rag.vectorstore import get_bm25_index, get_vectorstore

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def chunk_id(source: str, page, index: int, content: str) -> str:
    raw = f"{source}|{page}|{index}|{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def load_file_with_metadata(file_path: str, source_name: str | None = None):
    ext = os.path.splitext(file_path)[1].lower()
    source = source_name or os.path.basename(file_path)

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(file_path).load()
        for d in docs:
            # PyPDFLoader pages are 0-indexed; citations should match what readers see.
            page = d.metadata.get("page", d.metadata.get("page_number"))
            d.metadata = {"source": source, "page": int(page) + 1 if page is not None else None}
    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(file_path, autodetect_encoding=True).load()
        for d in docs:
            d.metadata = {"source": source, "page": 1}
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Chroma rejects None metadata values.
    for d in docs:
        d.metadata = {k: v for k, v in d.metadata.items() if v is not None}
    return docs


def delete_source(source: str) -> None:
    """Remove all chunks of a source from both indexes (used to replace re-uploaded files)."""
    vs = get_vectorstore()
    try:
        existing = vs.get(where={"source": source}).get("ids", [])
        if existing:
            vs.delete(ids=existing)
    except Exception:
        logger.exception("Could not delete old Chroma chunks for %s", source)
    get_bm25_index().delete_source(source)


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
