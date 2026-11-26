"""Multi-turn support: rewrite a follow-up question into a standalone one.

"What about international orders?" after a question about refunds becomes
"What is the refund window for international orders?", so retrieval, the cache
and the answer checks all work on a self-contained question.
"""
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from halo.core.llm import get_llm
from halo.core.log import get_logger

logger = get_logger(__name__)

MAX_TURNS = 6          # most recent messages used as context
MAX_TURN_CHARS = 1500  # per message, so long answers don't crowd the prompt

_SYSTEM = """You rewrite the user's latest message into a standalone question.
Use the conversation only to resolve references (it, that, they, the second one, what about...).
Keep the user's wording, names and numbers. Do not answer the question.
If the latest message is already standalone, return it unchanged. Return only the question."""


def _transcript(history: List[Dict[str, str]]) -> str:
    lines = []
    for turn in history[-MAX_TURNS:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:MAX_TURN_CHARS]}")
    return "\n".join(lines)


def rewrite_standalone(question: str, transcript: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "Conversation:\n{transcript}\n\nLatest message: {question}\n\nStandalone question:"),
    ])
    return (prompt | get_llm(0.0, task="fast") | StrOutputParser()).invoke({"transcript": transcript, "question": question})


def condense_question(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Return a standalone version of `question`. No history → no LLM call."""
    transcript = _transcript(history or [])
    if not transcript:
        return question
    try:
        standalone = (rewrite_standalone(question, transcript) or "").strip().strip('"')
    except Exception:
        logger.exception("condense: rewrite failed, using the message as-is")
        return question
    if not standalone or len(standalone) > 2000:
        return question
    if standalone != question:
        logger.info("condense: %r -> %r", question, standalone)
    return standalone
