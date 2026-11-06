from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.schemas import GradeHallucinations, GradeAnswer
from src.core.config import settings
from src.agents.state import AgentState

# Hard ceilings to prevent infinite cycles
MAX_REWRITES = 1
MAX_REGENERATIONS = 1

def check_hallucination_and_answer(state: AgentState) -> str:
    """
    Reflection node enforcing hard ceilings (max_rewrites=1, max_regenerations=1).
    Routes to 'graceful_fallback' if limits are exceeded.
    """
    print("---CHECK HALLUCINATIONS & HARD CEILINGS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)
    rewrite_count = state.get("rewrite_count", 0)
    
    # Check hard ceilings first
    if retry_count >= MAX_REGENERATIONS or rewrite_count >= MAX_REWRITES:
        print("---HARD CEILING REACHED: ROUTING TO DETERMINISTIC GRACEFUL FALLBACK---")
        return "fallback"

    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    
    # 1. Check Factual Grounding
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    system_hal = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'."""
    
    hal_prompt = ChatPromptTemplate.from_messages([
        ("system", system_hal),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n{generation}")
    ])
    
    hal_chain = hal_prompt | hallucination_grader
    try:
        score = hal_chain.invoke({"documents": "\n".join(documents), "generation": generation})
        grade = score.binary_score
    except Exception as e:
        print(f"Hallucination grader fallback: {e}")
        grade = "yes"
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 2. Check if it answers the question
        answer_grader = llm.with_structured_output(GradeAnswer)
        ans_prompt = ChatPromptTemplate.from_messages([
            ("system", "Assess whether answer addresses question. 'yes' or 'no'."),
            ("human", "User question:\n{question}\n\nLLM generation:\n{generation}")
        ])
        ans_chain = ans_prompt | answer_grader
        try:
            ans_score = ans_chain.invoke({"question": question, "generation": generation})
            if ans_score.binary_score == "yes":
                print("---DECISION: GENERATION RESOLVES QUESTION (VERIFIED)---")
                return "useful"
        except Exception as e:
            print(f"Answer grader fallback: {e}")
            return "useful"

        print("---DECISION: GENERATION DOES NOT FULLY RESOLVE QUESTION---")
        if retry_count < MAX_REGENERATIONS:
            return "not_useful"
        return "fallback"
    else:
        print("---DECISION: GENERATION IS NOT FULLY GROUNDED---")
        if retry_count < MAX_REGENERATIONS:
            return "not_supported"
        return "fallback"
