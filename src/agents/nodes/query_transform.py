from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import AgentState
from src.core.llm import get_llm
from src.core.log import get_logger

logger = get_logger(__name__)

_SYSTEM = """You rewrite questions into concise, well-formed web search queries.
Keep the user's intent, names, numbers and constraints. Return only the query text."""


def rewrite_query(question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "Question: {question}\n\nSearch query:"),
    ])
    return (prompt | get_llm(0.0) | StrOutputParser()).invoke({"question": question}).strip()


def transform_query(state: AgentState) -> dict:
    """Produce a web-search query. The user's `question` is left untouched so the
    answer and the answer check are always judged against what was actually asked."""
    question = state["question"]
    try:
        search_query = rewrite_query(question) or question
    except Exception:
        logger.exception("transform: rewrite failed, using original question")
        search_query = question

    logger.info("transform: %r", search_query)
    return {"search_query": search_query, "rewrite_count": state.get("rewrite_count", 0) + 1}
