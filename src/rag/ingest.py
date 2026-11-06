import os
import argparse
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.vectorstore import get_vectorstore, get_bm25_index

def load_file_with_metadata(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)
    
    docs = []
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        for d in raw_docs:
            d.metadata["source"] = base_name
            # Preserve page numbers explicitly
            if "page" not in d.metadata and "page_number" in d.metadata:
                d.metadata["page"] = d.metadata["page_number"]
            docs.append(d)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, autodetect_encoding=True)
        raw_docs = loader.load()
        for d in raw_docs:
            d.metadata["source"] = base_name
            d.metadata["page"] = 1
            docs.append(d)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return docs

def ingest_file_list(file_paths: List[str]) -> Tuple[int, int]:
    """
    Ingests documents into both the Dense ChromaDB vectorstore AND the BM25 Lexical index,
    ensuring page numbers, source provenance, and metadata are intact.
    """
    docs = []
    for path in file_paths:
        try:
            loaded = load_file_with_metadata(path)
            docs.extend(loaded)
        except Exception as e:
            print(f"Warning: Failed to load file {path}: {e}")
            
    if not docs:
        return 0, 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    if splits:
        # 1. Add to Chroma Dense Vector Store
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=splits)

        # 2. Add to BM25 Lexical Index
        bm25_index = get_bm25_index()
        split_texts = [s.page_content for s in splits]
        split_metas = [s.metadata for s in splits]
        bm25_index.add_documents(documents=split_texts, metadatas=split_metas)
        
    return len(docs), len(splits)

def ingest_documents(source_dir: str) -> Tuple[int, int]:
    print(f"Loading documents from {source_dir}...")
    
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith((".pdf", ".txt", ".md")):
                file_paths.append(os.path.join(root, f))
                
    if not file_paths:
        print("No supported documents found (.pdf, .txt, .md)")
        return 0, 0
        
    return ingest_file_list(file_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents to hybrid vector + BM25 store")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents")
    args = parser.parse_args()
    num_docs, num_chunks = ingest_documents(args.dir)
    print(f"Ingested {num_docs} documents into {num_chunks} chunks.")
