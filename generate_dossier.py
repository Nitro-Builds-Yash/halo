import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Don't draw on cover page (page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 11 * inch - 36, "HALO — Agentic Corrective RAG (CRAG) System Architecture & Technical Interview Dossier")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
            
            # Footer
            self.setFont("Helvetica", 8)
            self.drawString(54, 36, "Confidential — System Architecture & Interview Preparation Guide")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 54, 36, page_text)
            self.line(54, 46, 8.5 * inch - 54, 46)
            
        self.restoreState()

def create_halo_dossier(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0f172a") # Slate 900
    accent_color = colors.HexColor("#0284c7")  # Sky 600
    navy_color = colors.HexColor("#1e3a8a")    # Blue 900
    dark_gray = colors.HexColor("#1e293b")
    text_color = colors.HexColor("#334155")
    code_bg = colors.HexColor("#0f172a")
    code_text = colors.HexColor("#38bdf8")

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=primary_color,
        alignment=0,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=accent_color,
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=navy_color,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=accent_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=text_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.8,
        leading=10.8,
        textColor=colors.HexColor("#f8fafc"),
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0369a1")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=text_color
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.white
    )

    story = []

    # ================= COVER / HEADER SECTION =================
    story.append(Paragraph("HALO: AGENTIC CORRECTIVE RAG (CRAG)", title_style))
    story.append(Paragraph("System Architecture, LangGraph Cyclic State Orchestration, Core Implementation & Comprehensive Interview Dossier", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceAfter=18))

    meta_data = [
        [Paragraph("<b>Project:</b> Halo Autonomous CRAG System", table_cell), Paragraph("<b>Author:</b> Athota Yashwanth (Nitro-Builds-Yash)", table_cell)],
        [Paragraph("<b>Core Stack:</b> LangGraph, ChromaDB, FastAPI, SSE, Streamlit", table_cell), Paragraph("<b>Architecture:</b> Corrective Cyclic Multi-Agent RAG", table_cell)],
        [Paragraph("<b>Document Scope:</b> Comprehensive System Explanation + Full Code Snippets + Q&A", table_cell), Paragraph("<b>Target Role:</b> Senior AI/ML Engineer, LLM Systems Architect", table_cell)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # ================= EXECUTIVE SUMMARY =================
    story.append(Paragraph("1. Executive Overview & Problem Statement", h1_style))
    story.append(Paragraph(
        "Standard Retrieval-Augmented Generation (Naive RAG) is fundamentally <b>feed-forward and passive</b>: "
        "User Query &rarr; Vector Search &rarr; Top-K Context Injection &rarr; LLM Generation. "
        "This naive approach fails frequently in production due to three core vulnerabilities: "
        "<b>1) Semantic Mismatch:</b> Vector distance retrieves superficially similar text that lacks the actual answer; "
        "<b>2) Hallucination & Fact Drift:</b> The LLM fills informational gaps with unverified internal weights; "
        "<b>3) Silent Degradation:</b> The pipeline cannot audit whether the retrieved chunks actually answer the query, nor can it retry or fall back.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Halo solves this by implementing an Agentic Corrective RAG (CRAG) architecture built with LangGraph.</b> "
        "Instead of hoping retrieved documents are accurate, Halo treats retrieval as an iterative hypothesis. "
        "It features <b>autonomous quality gates</b>: it grades document relevance using structured LLM schemas, "
        "triggers query rewriting and external web search fallback when local corpus relevance fails, generates strictly bounded responses, "
        "and runs reflective hallucination checks before releasing answers to the client.",
        body_style
    ))

    # ================= ARCHITECTURAL WORKFLOW =================
    story.append(Paragraph("2. Architectural Blueprint & LangGraph Cyclic Workflow", h1_style))
    story.append(Paragraph(
        "The core of Halo is modeled as a <b>StateGraph</b> with cyclic conditional branches. Below is the precise operational progression:",
        body_style
    ))

    workflow_data = [
        [Paragraph("Step", table_header), Paragraph("Node / Gate", table_header), Paragraph("Decision Logic / Mechanism", table_header), Paragraph("Outcome", table_header)],
        [
            Paragraph("1", table_cell),
            Paragraph("<b>retrieve</b>", table_cell),
            Paragraph("Embeds query via OpenAI text-embedding and queries local ChromaDB collections with Top-K constraint.", table_cell),
            Paragraph("Populates state['documents'].", table_cell)
        ],
        [
            Paragraph("2", table_cell),
            Paragraph("<b>grade_documents</b>", table_cell),
            Paragraph("Evaluates every chunk via structured schema <code>GradeDocuments</code> (binary 'yes'/'no').", table_cell),
            Paragraph("Retains valid chunks; sets <code>web_search_needed=True</code> if corpus is irrelevant.", table_cell)
        ],
        [
            Paragraph("3", table_cell),
            Paragraph("<b>decide_to_generate</b><br/><i>(Conditional Edge)</i>", table_cell),
            Paragraph("Checks <code>state['web_search_needed']</code>. If False, proceeds to <b>generate</b>. If True, redirects to <b>transform_query</b>.", table_cell),
            Paragraph("Routing fork between immediate synthesis vs fallback.", table_cell)
        ],
        [
            Paragraph("4", table_cell),
            Paragraph("<b>transform_query</b>", table_cell),
            Paragraph("LLM rewrites input query, stripping conversational noise to optimize for keyword/semantic web indexing.", table_cell),
            Paragraph("Updates <code>state['question']</code> with enhanced search query.", table_cell)
        ],
        [
            Paragraph("5", table_cell),
            Paragraph("<b>web_search</b>", table_cell),
            Paragraph("Executes live query using Tavily API (with automatic fallback to DuckDuckGo search).", table_cell),
            Paragraph("Appends live external web facts into <code>state['documents']</code>.", table_cell)
        ],
        [
            Paragraph("6", table_cell),
            Paragraph("<b>generate</b>", table_cell),
            Paragraph("Prompt-bounded synthesis using combined local & web evidence context.", table_cell),
            Paragraph("Populates <code>state['generation']</code>.", table_cell)
        ],
        [
            Paragraph("7", table_cell),
            Paragraph("<b>check_hallucination_and_answer</b><br/><i>(Reflection Gate)</i>", table_cell),
            Paragraph("Dual structured audit: 1) Is generation grounded in evidence? (<code>GradeHallucinations</code>); 2) Does generation resolve user question? (<code>GradeAnswer</code>).", table_cell),
            Paragraph("If verified &rarr; <b>END</b>.<br/>If ungrounded &rarr; <b>increment_retry &rarr; transform_query</b>.", table_cell)
        ]
    ]

    wf_table = Table(workflow_data, colWidths=[28, 88, 230, 158])
    wf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(wf_table)
    story.append(Spacer(1, 15))

    # ================= CORE CODE IMPLEMENTATION =================
    story.append(PageBreak())
    story.append(Paragraph("3. Core Technical Implementation & Code Walkthrough", h1_style))
    story.append(Paragraph(
        "In technical interviews, interviewers expect you to explain how state transitions, prompt chaining, "
        "and type schemas work in code. Below are the key modules powering Halo.",
        body_style
    ))

    # --- 3.1 State Schema ---
    story.append(Paragraph("3.1 Agent Shared State Definition (src/agents/state.py)", h2_style))
    story.append(Paragraph(
        "Halo uses a typed dictionary (<code>TypedDict</code>) that flows through every node in the LangGraph execution graph:",
        body_style
    ))

    state_code = """from typing import TypedDict, List

class AgentState(TypedDict):
    question: str            # Original or rewritten question
    documents: List[str]     # Chunks retained from vectorstore / web search
    web_search_needed: bool  # Flag set when local chunks fail relevance check
    generation: str          # Current generated answer draft
    retry_count: int         # Current retry count to prevent infinite recursion"""
    
    c_table = Table([[Paragraph(state_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)]], colWidths=[504])
    c_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(c_table)
    story.append(Spacer(1, 10))

    # --- 3.2 Graph Construction ---
    story.append(Paragraph("3.2 StateGraph & Cyclic Routing Construction (src/agents/graph.py)", h2_style))
    story.append(Paragraph(
        "This defines the topological DAG with conditional routing and self-corrective cycles:",
        body_style
    ))

    graph_code = """from langgraph.graph import END, StateGraph
from src.agents.state import AgentState
from src.agents.nodes.retrieval import retrieve
from src.agents.nodes.grading import grade_documents
from src.agents.nodes.query_transform import transform_query
from src.agents.nodes.web_search import web_search
from src.agents.nodes.generation import generate
from src.agents.nodes.reflection import check_hallucination_and_answer

def decide_to_generate(state: AgentState):
    if state["web_search_needed"]:
        return "transform_query"
    return "generate"

def increment_retry(state: AgentState):
    return {"retry_count": state.get("retry_count", 0) + 1}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("increment_retry", increment_retry)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {
        "transform_query": "transform_query",
        "generate": "generate",
    })
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges("generate", check_hallucination_and_answer, {
        "not_supported": "increment_retry",
        "not_useful": "increment_retry",
        "useful": END,
        "max_retries": END
    })
    workflow.add_edge("increment_retry", "transform_query")
    return workflow.compile()"""

    g_table = Table([[Paragraph(graph_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)]], colWidths=[504])
    g_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(g_table)
    story.append(Spacer(1, 10))

    # --- 3.3 Grading Node ---
    story.append(PageBreak())
    story.append(Paragraph("3.3 Relevance Grader with Structured Pydantic Output (src/agents/nodes/grading.py)", h2_style))
    story.append(Paragraph(
        "Rather than parsing loose string outputs with regex, Halo uses OpenAI / LangChain structured outputs enforcing strict binary schema:",
        body_style
    ))

    grading_code = """from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to question, 'yes' or 'no'"
    )

def grade_documents(state: AgentState) -> dict:
    question = state["question"]
    documents = state["documents"]
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    structured_grader = llm.with_structured_output(GradeDocuments)
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "Assess relevance of retrieved document. Binary 'yes'/'no'."),
        ("human", "Retrieved document:\\n{document}\\n\\nQuestion: {question}")
    ])
    retrieval_grader = grade_prompt | structured_grader

    filtered_docs = []
    web_search_needed = False
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc})
        if score.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            web_search_needed = True
            
    if len(filtered_docs) == 0:
        web_search_needed = True
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}"""

    gr_table = Table([[Paragraph(grading_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)]], colWidths=[504])
    gr_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(gr_table)
    story.append(Spacer(1, 10))

    # --- 3.4 Reflection & Hallucination Auditor ---
    story.append(Paragraph("3.4 Reflection & Hallucination Auditor (src/agents/nodes/reflection.py)", h2_style))
    story.append(Paragraph(
        "Halo performs a two-tier evaluation on the candidate generation: Factual Grounding followed by Query Resolution.",
        body_style
    ))

    reflection_code = '''def check_hallucination_and_answer(state: AgentState) -> str:
    question, documents, generation = state["question"], state["documents"], state["generation"]
    retry_count = state.get("retry_count", 0)
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)

    # Gate 1: Check factual grounding against context
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hal_score = hallucination_grader.invoke({
        "documents": "\\n".join(documents), "generation": generation
    })
    
    if hal_score.binary_score == "yes":
        # Gate 2: Check if answer resolves user question
        answer_grader = llm.with_structured_output(GradeAnswer)
        ans_score = answer_grader.invoke({"question": question, "generation": generation})
        if ans_score.binary_score == "yes":
            return "useful"       # Verified -> END
        return "not_useful" if retry_count < settings.MAX_RETRIES else "max_retries"
    return "not_supported" if retry_count < settings.MAX_RETRIES else "max_retries"'''

    rf_table = Table([[Paragraph(reflection_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)]], colWidths=[504])
    rf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(rf_table)
    story.append(Spacer(1, 10))

    # --- 3.5 Streaming API ---
    story.append(PageBreak())
    story.append(Paragraph("3.5 Production Streaming & Document Ingestion API (src/api/server.py)", h2_style))
    story.append(Paragraph(
        "The server exposes Server-Sent Events (SSE) so client UIs can visualize node state changes as the agent thinks, "
        "as well as multipart file upload endpoints for dynamic indexing:",
        body_style
    ))

    api_code = """@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request, query: QueryRequest):
    async def event_generator():
        try:
            inputs = {"question": query.question, "retry_count": 0}
            async for output in graph.astream(inputs):
                for node_name, state_update in output.items():
                    # Stream intermediate node transitions to client
                    yield {
                        "event": "node_change",
                        "data": json.dumps({"node": node_name})
                    }
                    if "generation" in state_update and node_name == "generate":
                        yield {
                            "event": "message",
                            "data": json.dumps({"content": state_update["generation"]})
                        }
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
    return EventSourceResponse(event_generator())"""

    api_table = Table([[Paragraph(api_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)]], colWidths=[504])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(api_table)
    story.append(Spacer(1, 15))

    # ================= TOP 10 INTERVIEW QUESTIONS & ANSWERS =================
    story.append(Paragraph("4. Top 10 Technical Interview Questions & Answers", h1_style))
    story.append(Paragraph(
        "These questions represent deep-dive technical evaluations commonly encountered for Senior AI Engineer, "
        "LLM Architect, and RAG Specialist positions.",
        body_style
    ))

    qa_list = [
        ("Q1: What is the fundamental difference between Naive RAG, Advanced RAG, and Agentic CRAG?",
         "<b>Naive RAG</b> is purely linear: retrieve Top-K chunks &rarr; concatenate &rarr; generate. It has zero fallback if retrieved chunks are irrelevant.<br/>"
         "<b>Advanced RAG</b> adds pre-retrieval (query expansion, re-ranking) and post-retrieval (compression), but remains a single-pass feed-forward pipeline.<br/>"
         "<b>Agentic Corrective RAG (CRAG)</b> (implemented in Halo) introduces <i>control flow, self-evaluation, and cyclicity</i>: the system inspects retrieved chunks, dynamically decides whether internal documents are sufficient, triggers external tools (web search) if needed, and critiques its own generated response before outputting."),

        ("Q2: Why use LangGraph instead of standard LangChain chains or sequential pipelines?",
         "Standard chains (SequentialChain/LCEL) are strictly Directed Acyclic Graphs (DAGs) without loops. In an agentic system that requires reflection and corrective retries, execution must <b>cycle backwards</b> based on runtime decisions (e.g., if generated text fails faithfulness, increment retry count and loop back to query transformation). LangGraph models state as an explicit state machine with conditional branching and cyclic graph edges."),

        ("Q3: How do you prevent infinite loops when an agent reflects and retries?",
         "Halo maintains an explicit counter <code>retry_count: int</code> in the shared <code>AgentState</code>. In <code>src/agents/nodes/reflection.py</code>, if the hallucination or answer grader outputs 'no', the conditional edge checks <code>if retry_count < settings.MAX_RETRIES</code>. If max retries is reached, the system breaks out to <code>END</code> and returns the best effort or failure explanation rather than cycling endlessly."),

        ("Q4: Why does Halo use Pydantic Structured Outputs for grading rather than regex or string parsing?",
         "Prompting an LLM with 'Output YES or NO' frequently breaks in production due to conversational pleasantries ('Sure, the answer is Yes!') or formatting variations. Halo uses <code>llm.with_structured_output(GradeDocuments)</code> which leverages OpenAI Function Calling / Tool Calling under the hood. The model is constrained at the token-generation level to return a valid JSON object matching the Pydantic schema, eliminating parsing exceptions."),

        ("Q5: What is the purpose of the 'transform_query' node before executing web search?",
         "User questions are often conversational, long-winded, or contain irrelevant preamble (e.g., 'Hey I was wondering if you could explain how X affects Y?'). Web search engines (Tavily/DuckDuckGo) yield significantly higher precision when queries are condensed into dense, semantic search keywords. The rewriter abstracts the core intent into search-optimized terms."),

        ("Q6: How does Halo handle Fallback Mechanisms when external search tools fail?",
         "In <code>src/tools/search.py</code> and <code>src/agents/nodes/web_search.py</code>, Halo includes a two-tier fallback: if the primary API key (<code>TAVILY_API_KEY</code>) is missing, it dynamically swaps to DuckDuckGo search. Furthermore, search executions are wrapped in try-catch blocks that format structured web snippets while appending safe error notes if external rate limits occur, preventing unhandled HTTP exceptions."),

        ("Q7: How is real-time user experience handled given that multi-agent grading adds latency?",
         "Multi-step agent pipelines have higher latency than single-turn LLM calls. Halo addresses this by using <b>Server-Sent Events (SSE)</b> via <code>sse-starlette</code> in FastAPI and live status widgets in Streamlit. The system streams <code>node_change</code> events as each agent executes (Retrieving &rarr; Grading &rarr; Web Search &rarr; Generation), giving the user transparent, real-time feedback."),

        ("Q8: How does Halo ensure document chunking quality during ingestion?",
         "In <code>src/rag/ingest.py</code>, documents are split using <code>RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)</code>. Overlap prevents loss of contextual continuity at chunk boundaries. Ingest handles PDFs via <code>PyPDFLoader</code> and Markdown/Text with automatic character encoding detection."),

        ("Q9: What vector database is used, and how would you scale it to millions of documents?",
         "Halo currently uses persistent <b>ChromaDB</b> (local on-disk vector store). For enterprise scale (millions of chunks with high QPS), ChromaDB can be swapped with a distributed vector engine such as Qdrant, Milvus, or pgvector on PostgreSQL with HNSW indexes, without changing the LangGraph agent logic."),

        ("Q10: What metrics would you track in production to monitor the health of this CRAG system?",
         "1) <b>Retrieval Hit Rate / Grade Pass Rate:</b> Ratio of retrieved chunks graded as relevant.<br/>"
         "2) <b>Fallback Web Search Frequency:</b> Percentage of queries triggering web search (indicates internal knowledge gaps).<br/>"
         "3) <b>Faithfulness Audit Failure Rate:</b> Frequency of generations rejected by the hallucination grader.<br/>"
         "4) <b>End-to-End Latency & Cost:</b> Total token usage across grader, rewriter, and generator nodes.")
    ]

    for q_title, q_ans in qa_list:
        qa_data = [
            [Paragraph(f"<b>{q_title}</b>", table_header)],
            [Paragraph(q_ans, table_cell)]
        ]
        qa_table = Table(qa_data, colWidths=[504])
        qa_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy_color),
            ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(qa_table)
        story.append(Spacer(1, 8))

    # ================= PROJECT METRICS & RESUME SUMMARY =================
    story.append(PageBreak())
    story.append(Paragraph("5. Resume Bullet Points & Project Impact Summary", h1_style))
    story.append(Paragraph(
        "Use these high-impact bullet points when describing this project on your resume or LinkedIn profile:",
        body_style
    ))

    resume_points = [
        "<b>Architected an Autonomous Corrective RAG (CRAG) QA System:</b> Built an end-to-end multi-agent pipeline using LangGraph, ChromaDB, and OpenAI, overcoming naive RAG failure modes through automated relevance grading and self-corrective cycles.",
        "<b>Implemented Dynamic Fallback & Reflection Gates:</b> Designed Pydantic-validated LLM quality gates that audit retrieved context and generated answers for hallucinations, automatically falling back to Tavily web search when local context is insufficient.",
        "<b>Developed Full-Stack Streaming Architecture:</b> Created asynchronous FastAPI REST endpoints with Server-Sent Events (SSE) and an interactive Streamlit dashboard for real-time visualization of agent state transitions and document ingestion.",
        "<b>Hardened Production Reliability:</b> Prevented infinite recursion loops with stateful retry limits, integrated dual-search fallback (Tavily & DuckDuckGo), and implemented unit tests using Pytest and FastAPI TestClient."
    ]

    for pt in resume_points:
        story.append(Paragraph(f"&bull; {pt}", bullet_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceAfter=15))
    story.append(Paragraph("End of Halo Architecture & Technical Dossier — Built by Athota Yashwanth", callout_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Dossier PDF successfully created at: {filename}")

if __name__ == "__main__":
    output_pdf = r"C:\Users\yashw\.gemini\antigravity\scratch\halo\HALO_CRAG_INTERVIEW_DOSSIER.pdf"
    create_halo_dossier(output_pdf)
