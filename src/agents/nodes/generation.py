from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agents.state import AgentState
from src.core.config import settings

def generate(state: AgentState) -> dict:
    """
    Generate answer using the vectorstore and web search results.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(documents)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nContext: {context}"),
        ("human", "{question}")
    ])
    
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": generation}
