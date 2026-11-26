from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from halo.agents.attribution import attribute
from halo.agents.schemas import AnswerWithCitations
from halo.agents.state import AgentState
from halo.core.config import settings
from halo.core.guard import UNTRUSTED_NOTE
from halo.core.llm import get_llm, structured_llm
from halo.core.log import get_logger

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I couldn't find information about this in the indexed documents or from web search."
)

_SYSTEM = """You are a rigorous QA assistant.
1. Answer the user question accurately and concisely, strictly based on the provided context.
2. If the context does not contain enough information, say exactly what is missing.
3. Attach citations (source, page, short verbatim snippet) for each factual claim. Only cite sources that appear in the context headers.
4. In `sentences`, repeat your answer split into sentences, in order and word for word, and give for each the [#n] numbers of the passages that directly support it. Use an empty list for a sentence no passage supports.
5. Give a confidence score between 0.0 and 1.0 for how completely and unambiguously the context answers the question:
   - 0.85 - 1.0: fully answered with direct evidence
   - 0.50 - 0.84: partially answered or inferred
   - 0.00 - 0.49: insufficient or speculative""" + "\n" + UNTRUSTED_NOTE


def format_context(documents: List[str], metas: List[Dict[str, Any]],
                   max_doc_chars: int | None = None, max_total_chars: int | None = None) -> str:
    """Render documents with source headers, truncating long documents and stopping at a
    total budget so prompts stay inside the model's context window. Documents arrive in
    relevance order, so the least relevant ones are dropped first."""
    max_doc = max_doc_chars or settings.MAX_DOC_CHARS
    budget = max_total_chars or settings.MAX_CONTEXT_CHARS
    blocks, used = [], 0
    for idx, doc in enumerate(documents):
        meta = metas[idx] if idx < len(metas) else {}
        page = meta.get("page")
        header = f"[#{idx + 1}] [Source: {meta.get('source', 'Unknown Source')} | Page: {page if page is not None else 'N/A'}]"
        body = doc if len(doc) <= max_doc else doc[:max_doc] + " [...]"
        block = f"{header}\n{body}"
        if blocks and used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def filter_citations(citations: List[Dict[str, Any]], metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop citations whose source wasn't in the context (models sometimes invent sources)
    and exact duplicates."""
    known = {str(m.get("source", "")).strip().lower() for m in metas if m.get("source")}
    seen, kept = set(), []
    for c in citations:
        source = str(c.get("source", "")).strip().lower()
        key = (source, c.get("page"), c.get("snippet"))
        if source in known and key not in seen:
            seen.add(key)
            kept.append(c)
    return kept


def generate_answer(question: str, context: str) -> AnswerWithCitations:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    return (prompt | structured_llm(AnswerWithCitations, temperature=None)).invoke(
        {"context": context, "question": question}
    )


def generate(state: AgentState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])
    metas = state.get("doc_metadata", [])

    if not documents:
        logger.info("generate: no context available")
        return {"generation": NO_CONTEXT_ANSWER, "confidence": 0.0, "citations": [], "sentences": []}

    context = format_context(documents, metas)
    try:
        res = generate_answer(question, context)
        labelled = [s.model_dump() for s in res.sentences]
        return {
            "generation": res.answer,
            "confidence": float(res.confidence),
            "citations": filter_citations([c.model_dump() for c in res.citations], metas),
            "sentences": attribute(res.answer, labelled, documents, metas),
        }
    except Exception:
        logger.exception("generate: structured output failed; falling back to plain text")
        raw = (ChatPromptTemplate.from_messages([
            ("system", "Answer the question based strictly on the context. If unknown, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]) | get_llm()).invoke({"context": context, "question": question})
        # Unstructured answers have no self-rated confidence; keep it below the cache threshold.
        text = getattr(raw, "content", str(raw))
        return {
            "generation": text,
            "confidence": 0.5,
            "citations": [],
            "sentences": attribute(text, None, documents, metas),  # no labels: match passages directly
        }
