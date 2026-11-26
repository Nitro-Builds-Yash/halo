import html
import os
import shutil
import sys
import tempfile

import streamlit as st

# Allow `streamlit run src/halo/ui/app.py` from a checkout without `pip install -e .`
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from halo.agents.conversation import condense_question
from halo.agents.graph import graph
from halo.agents.runner import cache_payload, cached_answer, should_cache, summarize
from halo.agents.state import initial_state
from halo.core.config import settings
from halo.core.semantic_cache import SemanticCache
from halo.core.tracing import run_config
from halo.rag.ingest import delete_source, ingest_file_list, list_sources

st.set_page_config(
    page_title="halo · verified answers from your documents",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Same visual language as the web app served at "/" (src/halo/web).
st.markdown("""
<style>
:root { --accent:#7c8cff; --accent-2:#a78bfa; --good:#3ecf8e; --warn:#f5b84a; --line:rgba(128,140,170,.22); }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 900px; padding-top: 2.2rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

.halo-brand { display:flex; align-items:center; gap:12px; margin-bottom:.4rem; }
.halo-logo { width:34px; height:34px; border-radius:50%; background:conic-gradient(from 200deg,var(--accent),var(--accent-2),var(--accent));
  display:grid; place-items:center; box-shadow:0 0 22px rgba(124,140,255,.25); }
.halo-logo::after { content:""; width:22px; height:22px; border-radius:50%; background:var(--background-color, #0e1117); }
.halo-name { font-weight:700; font-size:20px; line-height:1.1; }
.halo-tag { opacity:.6; font-size:12px; }

.halo-hero { text-align:center; padding:2.2rem 0 1.4rem; }
.halo-hero h1 { font-size:2.4rem; letter-spacing:-.03em; margin:0 0 .4rem; }
.halo-hero h1 em { font-style:normal; background:linear-gradient(135deg,var(--accent),var(--accent-2));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.halo-hero p { opacity:.72; max-width:560px; margin:0 auto; }

.step-log { display:inline-flex; align-items:center; gap:6px; margin:3px 0; padding:4px 12px; border-radius:999px;
  border:1px solid var(--line); border-left:3px solid var(--accent); font-size:13px; }

.citation-card { border:1px solid var(--line); border-radius:12px; padding:10px 14px; margin-bottom:8px; }
.citation-header { font-weight:600; font-size:13px; color:var(--accent); margin-bottom:3px; }
.citation-snippet { font-size:13px; opacity:.78; font-style:italic; }

[data-testid="stChatMessage"] { border-radius:14px; padding:.9rem 1rem; }
[data-testid="stChatInput"] textarea { font-size:15px; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,var(--accent),var(--accent-2)); border:0; }
[data-testid="stFileUploaderDropzone"] { border-radius:14px; border:1.5px dashed var(--line); }
[data-testid="stExpander"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers

STEP_LABELS = {
    "retrieve": "🔍 <b>Hybrid retrieval</b>: Chroma dense + BM25, fused with RRF, cross-encoder rerank",
    "grade_documents": "⚖️ <b>Relevance gate</b>: per-chunk cross-encoder scores + batched LLM grading",
    "transform_query": "🔄 <b>Query rewrite</b>: local context insufficient, preparing a web search",
    "web_search": "🌐 <b>Web search</b>: Tavily / DuckDuckGo",
    "generate": "✍️ <b>Answer synthesis</b>: answer with citations and confidence",
    "increment_retry": "⚠️ <b>Not verified</b>: retrying with web context",
    "graceful_fallback": "🛟 <b>Fallback</b>: returning best excerpts with a disclaimer",
}


def esc(value) -> str:
    """Escape text from documents, web pages or the LLM before putting it in HTML."""
    return html.escape("" if value is None else str(value))


def step(text_html: str, color: str | None = None):
    style = f' style="border-left-color: {color};"' if color else ""
    st.markdown(f'<div class="step-log"{style}>{text_html}</div>', unsafe_allow_html=True)


def render_confidence(confidence):
    if confidence is None:
        return
    conf = max(0.0, min(1.0, float(confidence)))
    col_bar, col_val = st.columns([4, 1])
    col_bar.progress(conf)
    col_val.caption(f"**Confidence:** {int(conf * 100)}%")


def render_citations(citations, expanded=False):
    if not citations:
        return
    with st.expander("📚 Source citations", expanded=expanded):
        for c in citations:
            page = c.get("page")
            st.markdown(
                f'<div class="citation-card">'
                f'<div class="citation-header">📄 {esc(c.get("source", "Unknown"))}'
                f'{f" (page {esc(page)})" if page is not None else ""}</div>'
                f'<div class="citation-snippet">"{esc(c.get("snippet", ""))}"</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_trace(meta):
    with st.expander("🔍 Quality & agent trace", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chunks used", meta.get("retained_chunks", 0))
        col2.metric("Web search", "Yes" if meta.get("web_search_used") else "No")
        col3.metric("Retries", meta.get("retries", 0))
        col4.metric("Verified", "Cached" if meta.get("cached") else ("No" if meta.get("fallback") else "Yes"))


def render_sentences(sentences, expanded=False):
    judged = [s for s in sentences or [] if s.get("supported") is not None]
    if not judged:
        return
    unsupported = sum(1 for s in judged if not s["supported"])
    label = (f"🧷 Sentence evidence: {unsupported} of {len(judged)} sentence(s) not found in the sources"
             if unsupported else f"🧷 Sentence evidence: all {len(judged)} sentences supported")
    with st.expander(label, expanded=expanded and unsupported > 0):
        for s in sentences:
            if s.get("supported") is None:
                continue
            mark = "✅" if s["supported"] else "⚠️"
            st.markdown(f"{mark} **{s['text']}**")
            for src in s.get("sources", []):
                page = f", page {src['page']}" if src.get("page") is not None else ""
                st.caption(f"{src.get('source')}{page}: “{src.get('snippet', '')}”")
            if not s["supported"]:
                st.caption("No passage in the sources supports this sentence.")


def render_answer(msg, expanded_citations=False):
    if msg.get("fallback"):
        st.warning("Partial answer: this could not be fully verified against the sources.")
    st.markdown(msg["content"])
    render_confidence(msg.get("confidence"))
    render_citations(msg.get("citations"), expanded=expanded_citations)
    render_sentences(msg.get("sentences"), expanded=expanded_citations)
    if "meta" in msg:
        render_trace(msg["meta"])


def run_pipeline(question: str, history: list | None = None) -> dict:
    """Run the graph with live step updates; returns the summarized result."""
    standalone = condense_question(question, history)
    if standalone != question:
        step(f"💬 <b>Follow-up understood as</b>: <i>{esc(standalone)}</i>")
    cache = SemanticCache.get_instance()
    hit = cached_answer(question, cache, standalone)
    if hit:
        step(f"⚡ <b>Semantic cache hit</b>: similarity {esc(hit['cached_similarity'])}")
        return hit

    final_state = initial_state(standalone)
    for output in graph.stream(final_state, config=run_config("ui")):
        for node, update in output.items():
            update = update or {}
            if node == "decompose_query":
                subs = update.get("sub_questions", [])
                step(f"🧩 <b>Query plan</b>: {len(subs)} sub-question(s): <i>{esc('; '.join(subs))}</i>")
            elif node == "reflect":
                step(f"🧪 <b>Verification</b>: {esc(update.get('verdict', ''))}",
                     None if update.get("verdict") == "useful" else "#f59e0b")
            elif node in STEP_LABELS:
                step(STEP_LABELS[node], "#ef4444" if node == "increment_retry" else None)
            final_state.update(update)

    result = summarize(question, final_state, standalone)
    if should_cache(result):
        cache.store(standalone, cache_payload(final_state))
    return result


def ingest_uploads(files):
    temp_dir = tempfile.mkdtemp(prefix="halo_st_")
    try:
        paths, names = [], []
        for i, f in enumerate(files):
            name = os.path.basename(f.name.replace("\\", "/"))
            path = os.path.join(temp_dir, f"{i}_{name}")
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
            names.append(name)
        n_docs, n_chunks = ingest_file_list(paths, names)
        SemanticCache.get_instance().clear()
        return n_docs, n_chunks
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown('''<div class="halo-brand"><span class="halo-logo"></span>
        <div><div class="halo-name">halo</div><div class="halo-tag">ask anything · retrieve truth</div></div></div>''',
                unsafe_allow_html=True)
    st.caption("Prefer a browser app? Run `halo serve` and open http://localhost:8000")
    st.markdown("#### Documents")
    uploaded_files = st.file_uploader("Upload files (PDF, TXT, MD)", type=["pdf", "txt", "md"],
                                      accept_multiple_files=True)
    if st.button("🚀 Ingest & index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one document first.")
        else:
            with st.spinner("Embedding and indexing (Chroma + BM25)..."):
                try:
                    n_docs, n_chunks = ingest_uploads(uploaded_files)
                    st.success(f"✅ Ingested {n_docs} page(s) into {n_chunks} chunks.")
                except Exception as e:
                    st.error(f"❌ Ingestion error: {e}")

    sources = list_sources()
    if sources:
        st.caption(f"{len(sources)} document(s) indexed")
        for doc in sources:
            col_name, col_del = st.columns([5, 1])
            col_name.markdown(f"📄 {esc(doc['source'])}")
            col_name.caption(f"{doc['chunks']} chunks")
            if col_del.button("🗑️", key=f"del_{doc['source']}", help=f"Remove {doc['source']} from the index"):
                delete_source(doc["source"])
                SemanticCache.get_instance().clear()
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    with st.expander("Pipeline settings"):
        from halo.rag.bm25 import backend_name

        st.caption(f"**Model** `{settings.LLM_MODEL}`  \n"
                   f"**Embeddings** `{settings.EMBEDDING_MODEL}`  \n"
                   f"**Keyword search** `{backend_name()}`  \n"
                   f"**Top-K chunks** `{settings.TOP_K_RETRIEVE}`  \n"
                   f"**Retries / web rewrites** `{settings.MAX_REGENERATIONS}` / `{settings.MAX_REWRITES}`")
    if not settings.llm_configured:
        st.error("Set OPENAI_API_KEY (or LLM_BASE_URL for a local model) in .env")

    if st.button("New conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('''<div class="halo-hero"><h1>Answers you can <em>verify</em>.</h1>
        <p>Ask about your documents. Every sentence comes back with the passage that supports it,
        and anything halo can't back up is flagged.</p></div>''', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg)
        else:
            st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about your policies, papers or manuals...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_box = st.status("⚡ Running the CRAG pipeline...", expanded=True)
        try:
            with status_box:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                result = run_pipeline(user_query, history)
            status_box.update(label="✅ Done", state="complete", expanded=False)
        except Exception as e:
            status_box.update(label="❌ Pipeline error", state="error", expanded=True)
            st.error(f"The question could not be processed: {e}")
            st.stop()

        msg = {
            "role": "assistant",
            "content": result["generation"] or "No answer produced.",
            "confidence": result["confidence"],
            "citations": result["citations"],
            "sentences": result.get("sentences", []),
            "fallback": result["is_fallback"],
            "meta": {
                "retained_chunks": len(result["documents"]),
                "web_search_used": result["web_search_used"],
                "retries": result["retry_count"],
                "fallback": result["is_fallback"],
                "cached": result["cached"],
            },
        }
        render_answer(msg, expanded_citations=True)
        st.session_state.messages.append(msg)
