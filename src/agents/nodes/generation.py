from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.schemas import AnswerWithCitations
from src.core.config import settings

def generate(state: AgentState) -> dict:
    """
    Synthesize a strictly grounded answer with structured citations,
    source page provenance, and calibrated numerical confidence score (0.0 to 1.0).
    """
    print("---GENERATE WITH CITATIONS & CONFIDENCE---")
    question = state["question"]
    documents = state.get("documents", [])
    doc_metadata = state.get("doc_metadata", [])

    # Format context with explicit index tags and source provenance
    context_blocks = []
    for idx, doc in enumerate(documents):
        meta = doc_metadata[idx] if idx < len(doc_metadata) else {}
        source = meta.get("source", "Unknown Source")
        page = meta.get("page", None)
        header = f"[Source: {source} | Page: {page if page is not None else 'N/A'}]"
        context_blocks.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
    structured_generator = llm.with_structured_output(AnswerWithCitations)

    system = """You are a rigorous, production-grade QA intelligence assistant.
Your instructions:
1. Answer the user question accurately and concisely, strictly based on the provided context.
2. If the context does not contain enough information, truthfully state what is missing.
3. Attach exact citations with the source filename, page number, and snippet for each factual claim.
4. Provide a realistic confidence score between 0.0 and 1.0 representing how completely and unambiguously the context answers the user query.
   - High (0.85 - 1.0): Fully answered with direct factual evidence in the context.
   - Medium (0.50 - 0.84): Partially answered or inferred from context.
   - Low (0.00 - 0.49): Insufficient evidence, speculative, or unknown."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    generator_chain = prompt | structured_generator

    try:
        res: AnswerWithCitations = generator_chain.invoke({"context": context, "question": question})
        citations_data = [c.model_dump() for c in res.citations]
        return {
            "generation": res.answer,
            "confidence": res.confidence,
            "citations": citations_data
        }
    except Exception as e:
        print(f"Structured generation fallback: {e}")
        # Graceful fallback to text parsing if structured generation hits a provider limitation
        raw_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based strictly on the context. If unknown, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        raw_chain = raw_prompt | llm
        raw_resp = raw_chain.invoke({"context": context, "question": question})
        return {
            "generation": raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp),
            "confidence": 0.75,
            "citations": []
        }
