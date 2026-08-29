from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agents.state import AgentState
from src.core.config import settings

def transform_query(state: AgentState) -> dict:
    """
    Transform the query to produce a better question for web search.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    
    system = """You are generating questions that are well optimized for retrieval. 
    Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])
    
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    
    return {"question": better_question}
