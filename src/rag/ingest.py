import os
import argparse
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.vectorstore import get_vectorstore

def ingest_documents(source_dir: str):
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
        return
        
    print(f"Loaded {len(docs)} documents. Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Created {len(splits)} chunks. Ingesting to Chroma...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    print("Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents to vectorstore")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents")
    args = parser.parse_args()
    ingest_documents(args.dir)
