from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from halo.agents.nodes.generation import format_context
from halo.agents.schemas import GradeAnswer, GradeHallucinations
from halo.agents.state import AgentState
from halo.core.config import settings
from halo.core.llm import get_llm
from halo.core.log import get_logger

logger = get_logger(__name__)

_GROUNDING_SYSTEM = """You are a grader assessing whether an LLM answer is grounded in / supported by a set of retrieved facts.
Answer 'yes' only if every factual claim in the answer is supported by the facts. Otherwise answer 'no'.
An answer that only says the information is not available counts as grounded."""

_ANSWER_SYSTEM = "Assess whether the answer addresses and resolves the user question. Give a binary score 'yes' or 'no'."


def check_grounding(context: str, generation: str) -> bool:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _GROUNDING_SYSTEM),
        ("human", "Set of facts:\n\n{documents}\n\nLLM answer:\n{generation}"),
    ])
    res = (prompt | get_llm(0.0).with_structured_output(GradeHallucinations)).invoke(
        {"documents": context, "generation": generation}
    )
    return res.binary_score == "yes"


def check_answers_question(question: str, generation: str) -> bool:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _ANSWER_SYSTEM),
        ("human", "User question:\n{question}\n\nLLM answer:\n{generation}"),
    ])
    res = (prompt | get_llm(0.0).with_structured_output(GradeAnswer)).invoke(
        {"question": question, "generation": generation}
    )
    return res.binary_score == "yes"


def calibrate_confidence(llm_confidence: float, doc_scores: List[Optional[float]]) -> float:
    """Blend the model's self-rated confidence with the best retrieval score, when one exists.
    Self-rated confidence alone is poorly calibrated."""
    llm_confidence = max(0.0, min(1.0, float(llm_confidence)))
    scored = [s for s in doc_scores if s is not None]
    if not scored:
        return round(llm_confidence, 3)
    return round(0.6 * llm_confidence + 0.4 * max(scored), 3)


def reflect(state: AgentState) -> dict:
    """Verify the latest answer. Runs on EVERY generation, including ones produced after a
    rewrite / web search. Fails closed: if a grader errors, the answer counts as unverified."""
    documents = state.get("documents", [])
    if not documents:
        return {"verdict": "no_context"}

    generation = state.get("generation", "")
    try:
        if not check_grounding(format_context(documents, state.get("doc_metadata", [])), generation):
            logger.info("reflect: answer NOT grounded")
            return {"verdict": "not_grounded"}
        if not check_answers_question(state["question"], generation):
            logger.info("reflect: answer does not resolve the question")
            return {"verdict": "not_useful"}
    except Exception:
        logger.exception("reflect: grader failed; treating answer as unverified")
        return {"verdict": "not_grounded"}

    logger.info("reflect: answer verified")
    return {
        "verdict": "useful",
        "confidence": calibrate_confidence(state.get("confidence", 0.0), state.get("doc_scores") or []),
    }


def route_after_reflect(state: AgentState) -> str:
    """Ceilings are checked AFTER grading: they only decide whether another attempt is allowed."""
    verdict = state.get("verdict")
    if verdict == "useful":
        return "useful"
    if (state.get("retry_count", 0) < settings.MAX_REGENERATIONS
            and state.get("rewrite_count", 0) < settings.MAX_REWRITES):
        return "retry"
    return "fallback"
