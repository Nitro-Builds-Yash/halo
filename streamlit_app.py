import os
import shutil
import sys
import tempfile

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph import graph
from src.agents.runner import cache_payload, cached_answer, should_cache, summarize, used_web_search
from src.agents.state import initial_state
from src.core.config import settings
from src.core.semantic_cache import SemanticCache
from src.rag.ingest import ingest_file_list

st.set_page_config(
    page_title="Halo — Production Agentic CRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #1e1b4b 100%);
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border: 1px solid #312e81;
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-tagline {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    .step-log {
        font-size: 0.88rem;
        padding: 8px 12px;
        border-left: 3px solid #6366f1;
        background: rgba(99, 102, 241, 0.08);
        margin: 5px 0;
        border-radius: 0 6px 6px 0;
    }
    .citation-card {
        padding: 10px 14px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .citation-header {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.85rem;
    }
    .citation-snippet {
        font-size: 0.82rem;
        color: #cbd5e1;
        font-style: italic;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ HALO: Production Agentic CRAG</div>
    <div class="hero-tagline">Hybrid RRF (Dense Chroma + BM25) • CrossEncoder Neural Rerank • Query Decomposition • Calibrated Confidence & Citations</div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Document Management & Settings
with st.sidebar:
    st.markdown("### 📂 Document Store (Hybrid Index)")
    st.caption("Upload files to synchronize Dense ChromaDB + BM25 Lexical Store.")
    
    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("🚀 Ingest & Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one document first.")
        else:
            with st.spinner("Building Dense Vectors + BM25 inverted index..."):
                temp_paths, source_names = [], []
                temp_dir = tempfile.mkdtemp(prefix="halo_st_")
                try:
                    for i, f in enumerate(uploaded_files):
                        name = os.path.basename(f.name.replace("\\", "/"))
                        path = os.path.join(temp_dir, f"{i}_{name}")
                        with open(path, "wb") as out:
                            out.write(f.getbuffer())
                        temp_paths.append(path)
                        source_names.append(name)

                    num_docs, num_chunks = ingest_file_list(temp_paths, source_names)
                    SemanticCache.get_instance().clear()
                    st.success(f"✅ Ingested {num_docs} page(s) into {num_chunks} hybrid chunks!")
                except Exception as e:
                    st.error(f"❌ Ingestion error: {e}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Specifications")
    st.info(f"**Model:** `{settings.LLM_MODEL}`\n\n"
            f"**Retrieval:** `Hybrid RRF (BM25 + Chroma)`\n\n"
            f"**Reranker:** `ms-marco-MiniLM-L-6-v2`\n\n"
            f"**Top-K Chunks:** `{settings.TOP_K_RETRIEVE}`\n\n"
            f"**Max Retries / Web Rewrites:** `{settings.MAX_REGENERATIONS}` / `{settings.MAX_REWRITES}`")
    
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
        
        # Display confidence bar if present
        if "confidence" in msg and msg["confidence"] is not None:
            conf = float(msg["confidence"])
            col_bar, col_val = st.columns([4, 1])
            with col_bar:
                st.progress(conf)
            with col_val:
                st.caption(f"**Confidence:** {int(conf * 100)}%")
                
        # Display citations if present
        if msg.get("citations"):
            with st.expander("📚 Grounded Source Citations", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">📄 {cit.get('source', 'Unknown')} (Page {cit.get('page', 'N/A')})</div>
                        <div class="citation-snippet">"{cit.get('snippet', '')}"</div>
                    </div>
                    """, unsafe_allow_html=True)

        if "meta" in msg:
            with st.expander("🔍 Quality & Agent Trace", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retained Chunks", msg["meta"].get("retained_chunks", 0))
                col2.metric("Fallback Search", "Yes" if msg["meta"].get("web_search_fallback") else "No")
                col3.metric("Reflection Retries", msg["meta"].get("reflection_retries", 0))

user_query = st.chat_input("Ask a deep question over policies, research papers, or manuals...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_box = st.status("⚡ Executing Agentic CRAG Pipeline...", expanded=True)
        cache = SemanticCache.get_instance()
        hit = cached_answer(user_query, cache)

        final_state = {}
        with status_box:
            if hit:
                st.markdown(f'<div class="step-log">⚡ <b>Semantic cache hit</b> — similarity {hit["cached_similarity"]}</div>', unsafe_allow_html=True)
                final_state = {**hit, "doc_metadata": []}
            for output in ([] if hit else graph.stream(initial_state(user_query))):
                for node_name, state_update in output.items():
                    if node_name == "decompose_query":
                        sub_qs = state_update.get("sub_questions", [])
                        st.markdown(f'<div class="step-log">🧩 <b>Step 1: Query Decomposition</b> — Decomposed into {len(sub_qs)} sub-query/queries: <i>{", ".join(sub_qs)}</i></div>', unsafe_allow_html=True)
                    elif node_name == "retrieve":
                        st.markdown('<div class="step-log">🔍 <b>Step 2: Hybrid Retrieval</b> — Fusing ChromaDB Dense Vectors + BM25 Lexical + CrossEncoder Rerank</div>', unsafe_allow_html=True)
                    elif node_name == "grade_documents":
                        st.markdown('<div class="step-log">⚖️ <b>Step 3: Relevance Grader</b> — Auditing chunks with structured schema</div>', unsafe_allow_html=True)
                    elif node_name == "transform_query":
                        st.markdown('<div class="step-log">🔄 <b>Step 4: Query Transform</b> — Reformulating query for search engine indexing</div>', unsafe_allow_html=True)
                    elif node_name == "web_search":
                        st.markdown('<div class="step-log">🌐 <b>Step 5: Web Search</b> — Querying Tavily/DuckDuckGo fallback tool</div>', unsafe_allow_html=True)
                    elif node_name == "generate":
                        st.markdown('<div class="step-log">✍️ <b>Step 6: Answer Synthesis</b> — Generating verified answer with confidence & citations</div>', unsafe_allow_html=True)
                    elif node_name == "reflect":
                        verdict = (state_update or {}).get("verdict", "")
                        st.markdown(f'<div class="step-log">🧪 <b>Verification</b> — grounding & answer check: <i>{verdict}</i></div>', unsafe_allow_html=True)
                    elif node_name == "graceful_fallback":
                        st.markdown('<div class="step-log" style="border-left-color: #f59e0b;">🛟 <b>Fallback</b> — returning best excerpts with a disclaimer</div>', unsafe_allow_html=True)
                    elif node_name == "increment_retry":
                        st.markdown('<div class="step-log" style="border-left-color: #ef4444;">⚠️ <b>Quality Gate Check:</b> Reflection triggered retry loop...</div>', unsafe_allow_html=True)
                    final_state.update(state_update or {})

        status_box.update(label="✅ Halo Pipeline Finished", state="complete", expanded=False)
        
        answer = final_state.get("generation", "No generation produced.")
        confidence = final_state.get("confidence", 0.0)
        if not hit:
            result = summarize(user_query, final_state)
            if should_cache(result):
                cache.store(user_query, cache_payload(final_state))
        citations = final_state.get("citations", [])

        st.markdown(answer)

        # Confidence Bar
        if confidence is not None:
            col_bar, col_val = st.columns([4, 1])
            with col_bar:
                st.progress(float(confidence))
            with col_val:
                st.caption(f"**Confidence:** {int(confidence * 100)}%")

        # Display Citations
        if citations:
            with st.expander("📚 Grounded Source Citations", expanded=True):
                for cit in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">📄 {cit.get('source', 'Unknown')} (Page {cit.get('page', 'N/A')})</div>
                        <div class="citation-snippet">"{cit.get('snippet', '')}"</div>
                    </div>
                    """, unsafe_allow_html=True)

        meta = {
            "retained_chunks": len(final_state.get("documents", [])),
            "web_search_fallback": hit.get("web_search_used", False) if hit else used_web_search(final_state),
            "reflection_retries": final_state.get("retry_count", 0),
            "sub_questions": final_state.get("sub_questions", [])
        }
        with st.expander("🔍 Quality & Agent Trace", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Retained Chunks", meta["retained_chunks"])
            col2.metric("Fallback Search", "Yes" if meta["web_search_fallback"] else "No")
            col3.metric("Reflection Retries", meta["reflection_retries"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "citations": citations,
            "meta": meta
        })
