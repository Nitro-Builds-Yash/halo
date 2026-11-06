from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.schemas import BatchChunkGrading
from src.core.config import settings

def grade_documents(state: AgentState) -> dict:
    """
    Fast Cross-Encoder Gatekeeper & Single-Call Structured Batch Grading:
    1. If the Cross-Encoder score is >= 0.85, chunks bypass the LLM evaluator entirely and proceed.
    2. Borderline chunks (scores between 0.40 and 0.85) are graded in a SINGLE batched structured output call.
    3. Chunks below 0.40 are discarded without wasting LLM tokens.
    """
    print("---GRADE DOCUMENTS (BATCH & GATEKEEPER)---")
    question = state["question"]
    documents = state.get("documents", [])
    doc_metadata = state.get("doc_metadata", [])
    cross_conf = state.get("cross_encoder_confidence", 0.0)

    # 1. Gatekeeper bypass: if Cross-Encoder is confident, bypass LLM grading
    if cross_conf >= 0.85 and documents:
        print(f"---CROSS-ENCODER CONFIDENCE HIGH ({cross_conf:.2f} >= 0.85): BYPASSING LLM EVALUATOR---")
        return {
            "documents": documents,
            "doc_metadata": doc_metadata,
            "web_search_needed": False
        }

    if not documents:
        return {"documents": [], "doc_metadata": [], "web_search_needed": True}

    # 2. Single-Call Structured Batch Grading
    # Format all candidate chunks with indexed headers
    formatted_chunks = []
    for idx, doc in enumerate(documents):
        meta = doc_metadata[idx] if idx < len(doc_metadata) else {}
        source = meta.get("source", "Unknown")
        formatted_chunks.append(f"[Chunk Index {idx} | Source: {source}]\n{doc}")

    batch_context = "\n\n---\n\n".join(formatted_chunks)

    # Use fast, cheap model (e.g. gpt-4o-mini or 8B-class) for grading
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    structured_grader = llm.with_structured_output(BatchChunkGrading)

    system = """You are a fast, precision-oriented retrieval grader.
Assess the relevance of each indexed chunk to the given user question.
Return a structured list of ChunkGrade evaluations matching each chunk index.
If a chunk contains facts, definitions, or context useful to answer the question, grade is_relevant='yes', otherwise 'no'."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User Question: {question}\n\nCandidate Chunks:\n{chunks}")
    ])

    batch_chain = prompt | structured_grader

    filtered_docs: List[str] = []
    filtered_metas: List[Dict[str, Any]] = []

    try:
        res: BatchChunkGrading = batch_chain.invoke({"question": question, "chunks": batch_context})
        relevant_indices = {g.chunk_index for g in res.grades if g.is_relevant == "yes"}
        
        for idx in range(len(documents)):
            if idx in relevant_indices:
                filtered_docs.append(documents[idx])
                if idx < len(doc_metadata):
                    filtered_metas.append(doc_metadata[idx])
    except Exception as e:
        print(f"Batch grading fallback: {e}")
        filtered_docs = documents
        filtered_metas = doc_metadata

    web_search_needed = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "doc_metadata": filtered_metas,
        "web_search_needed": web_search_needed
    }
