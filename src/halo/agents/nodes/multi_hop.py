"""Agentic multi-hop search.

Some questions need a fact that is only discoverable after reading another one ("Who leads
the team that built X, and when did they join?"). After the first retrieval, a planner model
reads compact notes of what was found and either says the notes are enough or asks for one
focused follow-up search. This repeats up to MAX_HOPS times. New chunks go through the same
de-duplication and prompt-injection guard as the first retrieval.
"""
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from halo.agents.nodes.retrieval import Collector
from halo.agents.schemas import HopDecision
from halo.agents.state import AgentState
from halo.core.config import settings
from halo.core.guard import UNTRUSTED_NOTE
from halo.core.llm import structured_llm
from halo.core.log import get_logger
from halo.rag.vectorstore import get_hybrid_retriever

logger = get_logger(__name__)

NOTE_CHARS = 350
MAX_NOTES = 10

_SYSTEM = """You plan document searches for a question-answering system.
Given the question and notes from passages found so far, decide whether the notes contain
everything needed to answer. If not, write ONE short, specific search query for the single most
important missing fact (use names, terms and numbers from the notes). Do not repeat earlier queries.
""" + UNTRUSTED_NOTE


def should_run(state: AgentState) -> bool:
    mode = (settings.AGENTIC_SEARCH or "auto").lower()
    if mode == "off" or settings.MAX_HOPS <= 0:
        return False
    if mode == "on":
        return True
    return len(state.get("sub_questions") or []) > 1


def _notes(documents: List[str], metas: List[Dict[str, Any]]) -> str:
    lines = []
    for i, (d, m) in enumerate(zip(documents[:MAX_NOTES], metas[:MAX_NOTES]), 1):
        text = re.sub(r"\s+", " ", d)[:NOTE_CHARS]
        lines.append(f"[{i}] ({m.get('source', '?')}) {text}")
    return "\n".join(lines) or "(nothing found yet)"


def plan_next(question: str, notes: str, previous: List[str]) -> HopDecision:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", "Question: {question}\n\nEarlier queries: {previous}\n\nNotes:\n{notes}"),
    ])
    return (prompt | structured_llm(HopDecision, task="fast")).invoke(
        {"question": question, "previous": "; ".join(previous) or "none", "notes": notes})


def multi_hop(state: AgentState) -> dict:
    if not should_run(state):
        return {}
    question = state["question"]
    collector = Collector(state.get("documents"), state.get("doc_metadata"), state.get("doc_scores"),
                          state.get("security_flags"))
    previous = list(state.get("sub_questions") or [question])
    hops: List[Dict[str, Any]] = list(state.get("hops") or [])
    retriever = get_hybrid_retriever()

    for _ in range(settings.MAX_HOPS):
        try:
            decision = plan_next(question, _notes(collector.docs, collector.metas), previous)
        except Exception:
            logger.exception("multi_hop: planner failed; continuing with what was found")
            break
        query = (decision.next_query or "").strip()
        if decision.enough or not query or query.lower() in {p.lower() for p in previous}:
            break
        added = collector.add(retriever.retrieve(query, top_k=settings.TOP_K_RETRIEVE, use_rerank=True))
        hops.append({"query": query, "missing": decision.missing, "new_chunks": added})
        previous.append(query)
        logger.info("multi_hop: %r -> %d new chunk(s)", query, added)
        if not added:
            break

    return {**collector.state(), "hops": hops}
