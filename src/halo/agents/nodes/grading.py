from typing import Any, Dict, List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate

from halo.agents.schemas import BatchChunkGrading
from halo.agents.state import AgentState
from halo.core.config import settings
from halo.core.llm import get_llm
from halo.core.log import get_logger

logger = get_logger(__name__)

_SYSTEM = """You are a fast, precision-oriented retrieval grader.
Assess the relevance of each indexed chunk to the given user question.
Return a structured list of ChunkGrade evaluations matching each chunk index.
If a chunk contains facts, definitions, or context useful to answer the question, grade is_relevant='yes', otherwise 'no'."""


def llm_batch_grade(question: str, chunks: List[str], metas: List[Dict[str, Any]]) -> Set[int]:
    """Grade chunks in ONE structured LLM call. Returns the indices (into `chunks`) judged relevant."""
    formatted = []
    for idx, doc in enumerate(chunks):
        source = (metas[idx] if idx < len(metas) else {}).get("source", "Unknown")
        formatted.append(f"[Chunk Index {idx} | Source: {source}]\n{doc}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "User Question: {question}\n\nCandidate Chunks:\n{chunks}"),
    ])
    chain = prompt | get_llm(0.0).with_structured_output(BatchChunkGrading)
    res: BatchChunkGrading = chain.invoke({"question": question, "chunks": "\n\n---\n\n".join(formatted)})
    return {g.chunk_index for g in res.grades if g.is_relevant == "yes" and 0 <= g.chunk_index < len(chunks)}


def grade_documents(state: AgentState) -> dict:
    """Per-chunk cross-encoder gatekeeper + single-call batch grading.

    - score >= GATE_HIGH  -> kept without an LLM call
    - score <  GATE_LOW   -> discarded without an LLM call
    - in between, or unscored (reranker unavailable) -> graded together in one LLM call
    """
    question = state["question"]
    documents = state.get("documents", [])
    metas = state.get("doc_metadata", [])
    scores: List[Optional[float]] = state.get("doc_scores") or [None] * len(documents)

    if not documents:
        logger.info("grade: no candidate chunks -> web search needed")
        return {"documents": [], "doc_metadata": [], "doc_scores": [], "web_search_needed": True}

    keep: Set[int] = set()
    borderline: List[int] = []
    for i in range(len(documents)):
        s = scores[i] if i < len(scores) else None
        if s is None:
            borderline.append(i)
        elif s >= settings.GATE_HIGH:
            keep.add(i)
        elif s >= settings.GATE_LOW:
            borderline.append(i)

    logger.info("grade: %d auto-kept, %d borderline, %d discarded",
                len(keep), len(borderline), len(documents) - len(keep) - len(borderline))

    if borderline:
        try:
            relevant = llm_batch_grade(
                question,
                [documents[i] for i in borderline],
                [metas[i] if i < len(metas) else {} for i in borderline],
            )
            keep.update(borderline[j] for j in relevant)
        except Exception:
            # Keep borderline context rather than lose it; the grounding check still guards the answer.
            logger.exception("grade: batch grading failed; keeping borderline chunks")
            keep.update(borderline)

    ordered = sorted(keep)
    return {
        "documents": [documents[i] for i in ordered],
        "doc_metadata": [metas[i] if i < len(metas) else {} for i in ordered],
        "doc_scores": [scores[i] if i < len(scores) else None for i in ordered],
        "web_search_needed": len(ordered) == 0,
    }
