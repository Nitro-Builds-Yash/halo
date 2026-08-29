from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.schemas import GradeHallucinations, GradeAnswer
from src.core.config import settings
from src.agents.state import AgentState

def check_hallucination_and_answer(state: AgentState) -> str:
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)
    
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    
    # 1. Check Hallucination
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    system_hal = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    
    hal_prompt = ChatPromptTemplate.from_messages([
        ("system", system_hal),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ])
    
    hal_chain = hal_prompt | hallucination_grader
    score = hal_chain.invoke({"documents": "\n".join(documents), "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 2. Check if it answers the question
        answer_grader = llm.with_structured_output(GradeAnswer)
        system_ans = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        ans_prompt = ChatPromptTemplate.from_messages([
            ("system", system_ans),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ])
        ans_chain = ans_prompt | answer_grader
        ans_score = ans_chain.invoke({"question": question, "generation": generation})
        
        if ans_score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if retry_count < settings.MAX_RETRIES:
                return "not_useful"
            return "max_retries"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        if retry_count < settings.MAX_RETRIES:
            return "not_supported"
        return "max_retries"
