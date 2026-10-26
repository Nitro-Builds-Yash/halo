from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

def get_vectorstore():
    # Allow passing explicit API key or fallback to environment
    api_key = settings.OPENAI_API_KEY if settings.OPENAI_API_KEY else "dummy-key-for-init"
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.TOP_K_RETRIEVE})
