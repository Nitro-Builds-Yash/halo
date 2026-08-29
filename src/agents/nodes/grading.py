from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.schemas import GradeDocuments
from src.core.config import settings

def grade_documents(state: AgentState) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    web_search_needed = False
    
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc})
        grade = score.binary_score
        
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            web_search_needed = True
            
    if len(filtered_docs) == 0:
        web_search_needed = True
        
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}
