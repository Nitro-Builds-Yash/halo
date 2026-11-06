from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings
from src.rag.bm25 import BM25Index
from src.rag.reranker import CrossEncoderReranker

_bm25_index_instance: Optional[BM25Index] = None

def get_bm25_index() -> BM25Index:
    global _bm25_index_instance
    if _bm25_index_instance is None:
        persist_file = f"{settings.CHROMA_PERSIST_DIR}/bm25_index.json"
        _bm25_index_instance = BM25Index(persist_path=persist_file)
    return _bm25_index_instance

def get_vectorstore():
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

class HybridRetriever:
    """
    Combines Dense Vector Retrieval (ChromaDB) with Lexical Search (BM25)
    fused via Reciprocal Rank Fusion (RRF) and re-scored with CrossEncoder Reranking.
    """
    def __init__(self, rrf_k: int = 60):
        self.vectorstore = get_vectorstore()
        self.bm25 = get_bm25_index()
        self.reranker = CrossEncoderReranker.get_instance()
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 4, use_rerank: bool = True) -> List[Dict[str, Any]]:
        # 1. Dense Vector Search
        dense_results: List[Dict[str, Any]] = []
        try:
            dense_docs = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k * 2)
            for doc, score in dense_docs:
                dense_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "dense_score": float(score)
                })
        except Exception as e:
            # Fallback if similarity_search_with_relevance_scores is unsupported
            try:
                raw_docs = self.vectorstore.similarity_search(query, k=top_k * 2)
                for doc in raw_docs:
                    dense_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "dense_score": 1.0
                    })
            except Exception as inner_e:
                print(f"Dense search notice: {inner_e}")

        # 2. BM25 Lexical Search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion (RRF)
        # score = 1 / (k + rank)
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            content = item["content"]
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if content not in doc_lookup:
                doc_lookup[content] = item

        for rank, item in enumerate(bm25_results):
            content = item["content"]
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if content not in doc_lookup:
                doc_lookup[content] = item

        fused_candidates: List[Dict[str, Any]] = []
        for content, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = dict(doc_lookup[content])
            entry["rrf_score"] = rrf_score
            fused_candidates.append(entry)

        # 4. Neural Cross-Encoder Reranking
        if use_rerank and fused_candidates:
            final_results = self.reranker.rerank(query, fused_candidates, top_k=top_k)
        else:
            final_results = fused_candidates[:top_k]

        return final_results
