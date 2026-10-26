import os
import argparse
from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.vectorstore import get_vectorstore

def load_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return loader.load()

def ingest_file_list(file_paths: List[str]) -> Tuple[int, int]:
    docs = []
    for path in file_paths:
        try:
            loaded = load_file(path)
            docs.extend(loaded)
        except Exception as e:
            print(f"Warning: Failed to load file {path}: {e}")
            
    if not docs:
        return 0, 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    if splits:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=splits)
        
    return len(docs), len(splits)

def ingest_documents(source_dir: str) -> Tuple[int, int]:
    print(f"Loading documents from {source_dir}...")
    
    text_loader_kwargs = {'autodetect_encoding': True}
    loaders = [
        DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs),
        DirectoryLoader(source_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs),
        DirectoryLoader(source_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    ]
    
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading with {loader}: {e}")
            
    if not docs:
        print("No documents found!")
        return 0, 0
        
    print(f"Loaded {len(docs)} documents. Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Created {len(splits)} chunks. Ingesting to Chroma...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    print("Ingestion complete!")
    return len(docs), len(splits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents to vectorstore")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents")
    args = parser.parse_args()
    ingest_documents(args.dir)
