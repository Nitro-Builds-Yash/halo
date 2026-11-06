from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.schemas import SubQueries
from src.core.config import settings

def decompose_query(state: AgentState) -> dict:
    """
    Analyzes whether the query is complex, comparative, or multi-hop,
    and decomposes it into distinct atomic sub-queries for precision retrieval.
    """
    print("---DECOMPOSE QUERY---")
    question = state["question"]
    
    # If the question is simple, keep it as single item
    words = question.strip().split()
    if len(words) <= 5 and not any(w in question.lower() for w in ["compare", "difference", "versus", "vs", "and", "both"]):
        return {"sub_questions": [question]}

    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    structured_decomposer = llm.with_structured_output(SubQueries)

    system = """You are an expert search query planner.
Given a user question, decompose it into 1 to 3 atomic sub-questions that must be answered to satisfy the user request.
For comparative or multi-hop questions, isolate each entity or temporal aspect into its own query.
If the question is already atomic, return a single sub-question identical to the original."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}")
    ])

    decomposer_chain = prompt | structured_decomposer
    try:
        res = decomposer_chain.invoke({"question": question})
        sub_qs = res.sub_questions if res.sub_questions else [question]
    except Exception as e:
        print(f"Query decomposition fallback: {e}")
        sub_qs = [question]

    return {"sub_questions": sub_qs}
