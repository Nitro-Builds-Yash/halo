import re

from langchain_core.prompts import ChatPromptTemplate

from halo.agents.schemas import SubQueries
from halo.agents.state import AgentState
from halo.core.llm import get_llm
from halo.core.log import get_logger

logger = get_logger(__name__)

# Whole-word match, so "standard" / "understand" no longer trigger decomposition via "and".
_COMPLEX_HINTS = re.compile(r"\b(compare|comparison|difference|differences|versus|vs|and|both|between)\b", re.I)

_SYSTEM = """You are an expert search query planner.
Given a user question, decompose it into 1 to 3 atomic sub-questions that must be answered to satisfy the user request.
For comparative or multi-hop questions, isolate each entity or temporal aspect into its own query.
If the question is already atomic, return a single sub-question identical to the original."""


def needs_decomposition(question: str) -> bool:
    return len(question.split()) > 12 or bool(_COMPLEX_HINTS.search(question))


def decompose_query(state: AgentState) -> dict:
    question = state["question"]
    if not needs_decomposition(question):
        return {"sub_questions": [question]}

    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", "Question: {question}")])
    chain = prompt | get_llm(0.0).with_structured_output(SubQueries)
    try:
        res = chain.invoke({"question": question})
        sub_qs = [q for q in (res.sub_questions or []) if q.strip()][:3] or [question]
    except Exception:
        logger.exception("decompose: failed, using original question")
        sub_qs = [question]

    logger.info("decompose: %s", sub_qs)
    return {"sub_questions": sub_qs}
