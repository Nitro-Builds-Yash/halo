from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from src.agents.schemas import AnswerWithCitations
from src.agents.state import AgentState
from src.core.llm import get_llm
from src.core.log import get_logger

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I couldn't find information about this in the indexed documents or from web search."
)

_SYSTEM = """You are a rigorous QA assistant.
1. Answer the user question accurately and concisely, strictly based on the provided context.
2. If the context does not contain enough information, say exactly what is missing.
3. Attach citations (source, page, short verbatim snippet) for each factual claim. Only cite sources that appear in the context headers.
4. Give a confidence score between 0.0 and 1.0 for how completely and unambiguously the context answers the question:
   - 0.85 - 1.0: fully answered with direct evidence
   - 0.50 - 0.84: partially answered or inferred
   - 0.00 - 0.49: insufficient or speculative"""


def format_context(documents: List[str], metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for idx, doc in enumerate(documents):
        meta = metas[idx] if idx < len(metas) else {}
        page = meta.get("page")
        header = f"[Source: {meta.get('source', 'Unknown Source')} | Page: {page if page is not None else 'N/A'}]"
        blocks.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, context: str) -> AnswerWithCitations:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    return (prompt | get_llm().with_structured_output(AnswerWithCitations)).invoke(
        {"context": context, "question": question}
    )


def generate(state: AgentState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])
    metas = state.get("doc_metadata", [])

    if not documents:
        logger.info("generate: no context available")
        return {"generation": NO_CONTEXT_ANSWER, "confidence": 0.0, "citations": []}

    context = format_context(documents, metas)
    try:
        res = generate_answer(question, context)
        return {
            "generation": res.answer,
            "confidence": float(res.confidence),
            "citations": [c.model_dump() for c in res.citations],
        }
    except Exception:
        logger.exception("generate: structured output failed; falling back to plain text")
        raw = (ChatPromptTemplate.from_messages([
            ("system", "Answer the question based strictly on the context. If unknown, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]) | get_llm()).invoke({"context": context, "question": question})
        # Unstructured answers have no self-rated confidence; keep it below the cache threshold.
        return {
            "generation": getattr(raw, "content", str(raw)),
            "confidence": 0.5,
            "citations": [],
        }
