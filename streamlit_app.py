import os
import sys
import tempfile
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph import graph
from src.rag.ingest import ingest_file_list
from src.core.config import settings

st.set_page_config(
    page_title="Halo — Agentic CRAG Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #38bdf8;
        margin: 0;
    }
    .hero-tagline {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        background-color: #0369a1;
        color: white;
    }
    .step-log {
        font-size: 0.88rem;
        padding: 6px 10px;
        border-left: 3px solid #38bdf8;
        background: rgba(56, 189, 248, 0.05);
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
    }
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ HALO — Agentic CRAG Assistant</div>
    <div class="hero-tagline">Self-corrective retrieval, relevance grading, hallucination auditing, and adaptive web search fallback.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Document Management & Settings
with st.sidebar:
    st.markdown("### 📂 Document Store")
    st.caption("Upload documents to build your local retrieval corpus.")
    
    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file first.")
        else:
            with st.spinner("Chunking & embedding documents..."):
                temp_paths = []
                temp_dir = tempfile.mkdtemp(prefix="halo_st_")
                try:
                    for f in uploaded_files:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.getbuffer())
                        temp_paths.append(path)
                    
                    num_docs, num_chunks = ingest_file_list(temp_paths)
                    st.success(f"✅ Ingested {num_docs} files into {num_chunks} vector chunks!")
                except Exception as e:
                    st.error(f"❌ Ingestion error: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Configuration")
    st.info(f"**Model:** `{settings.LLM_MODEL}`\n\n"
            f"**Top-K Retrieval:** `{settings.TOP_K_RETRIEVE}`\n\n"
            f"**Max Retries:** `{settings.MAX_RETRIES}`")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            with st.expander("🔍 Quality & Audit Trace", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retained Chunks", msg["meta"]["retained_chunks"])
                col2.metric("Fallback Search", "Yes" if msg["meta"]["web_search_fallback"] else "No")
                col3.metric("Reflections / Retries", msg["meta"]["reflection_retries"])
                
                if msg["meta"].get("sources"):
                    st.markdown("**Evidence Chunks:**")
                    for idx, src in enumerate(msg["meta"]["sources"]):
                        st.markdown(f"> *Source {idx+1}:* {src[:250]}...")

user_query = st.chat_input("Ask a question about your indexed knowledge base...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_box = st.status("⚡ Running Halo CRAG Pipeline...", expanded=True)
        inputs = {"question": user_query, "retry_count": 0}
        
        final_state = {}
        with status_box:
            for output in graph.stream(inputs):
                for node_name, state_update in output.items():
                    if node_name == "retrieve":
                        st.markdown('<div class="step-log">🔍 <b>Step 1: Retrieve</b> — Fetching semantic vectors from ChromaDB</div>', unsafe_allow_html=True)
                    elif node_name == "grade_documents":
                        st.markdown('<div class="step-log">⚖️ <b>Step 2: Grade Documents</b> — Auditing chunk relevance against query</div>', unsafe_allow_html=True)
                    elif node_name == "transform_query":
                        st.markdown('<div class="step-log">🔄 <b>Step 3: Transform Query</b> — Re-formulating search query for broader coverage</div>', unsafe_allow_html=True)
                    elif node_name == "web_search":
                        st.markdown('<div class="step-log">🌐 <b>Step 4: Web Search</b> — Querying Tavily/DuckDuckGo fallback tool</div>', unsafe_allow_html=True)
                    elif node_name == "generate":
                        st.markdown('<div class="step-log">✍️ <b>Step 5: Synthesize Answer</b> — Generating verified answer</div>', unsafe_allow_html=True)
                    elif node_name == "increment_retry":
                        st.markdown('<div class="step-log" style="border-left-color: #ef4444;">⚠️ <b>Quality Gate Check:</b> Hallucination detected or question unresolved; triggering reflection retry...</div>', unsafe_allow_html=True)
                    final_state.update(state_update)

        status_box.update(label="✅ Halo CRAG Pipeline Finished", state="complete", expanded=False)
        
        answer = final_state.get("generation", "No generation produced.")
        st.markdown(answer)

        meta = {
            "retained_chunks": len(final_state.get("documents", [])),
            "web_search_fallback": final_state.get("web_search_needed", False),
            "reflection_retries": final_state.get("retry_count", 0),
            "sources": final_state.get("documents", [])
        }
        with st.expander("🔍 Quality & Audit Trace", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Retained Chunks", meta["retained_chunks"])
            col2.metric("Fallback Search", "Yes" if meta["web_search_fallback"] else "No")
            col3.metric("Reflections / Retries", meta["reflection_retries"])
            if meta.get("sources"):
                st.markdown("**Evidence Chunks:**")
                for idx, src in enumerate(meta["sources"]):
                    st.markdown(f"> *Source {idx+1}:* {src[:250]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
