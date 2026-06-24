"""Streamlit UI for RAG with Cross-Encoder Reranking."""

import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import get_embedding_model
from src.ingestor import ingest_single_pdf, remove_pdf, remove_all_pdfs, PDF_STORE
from src.pipeline import RAGPipeline
from src.vectorstore import load_vectorstore

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="RAG · Cross-Encoder Reranking",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "FAISS_INDEX_PATH":    os.getenv("FAISS_INDEX_PATH",    "faiss_index"),
    "EMBEDDING_MODEL":     os.getenv("EMBEDDING_MODEL",     "all-MiniLM-L6-v2"),
    "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "TOP_K":               os.getenv("TOP_K",               "10"),
    "TOP_N":               os.getenv("TOP_N",               "3"),
    "CHUNK_SIZE":          os.getenv("CHUNK_SIZE",          "500"),
    "CHUNK_OVERLAP":       os.getenv("CHUNK_OVERLAP",       "50"),
    "LLM_PROVIDER":        os.getenv("LLM_PROVIDER",        "groq"),
    "LLM_MODEL":           os.getenv("LLM_MODEL",           "llama3-8b-8192"),
    "LLM_API_KEY":         os.getenv("LLM_API_KEY",         ""),
}

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "vectorstore":        None,
    "chat_history":       [],
    "db_ready":           False,
    # doc_registry: { filename: { "chunks": int, "status": str } }
    "doc_registry":       {},
    # confirmation flags
    "confirm_remove":     None,   # filename pending removal confirmation
    "confirm_remove_all": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def total_chunks() -> int:
    return sum(v["chunks"] for v in st.session_state.doc_registry.values())


def total_docs() -> int:
    return len(st.session_state.doc_registry)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return get_embedding_model(CONFIG["EMBEDDING_MODEL"])


# ---------------------------------------------------------------------------
# On startup: restore any existing index + registry from PDF_STORE
# ---------------------------------------------------------------------------
def _restore_on_startup():
    index_file = Path(CONFIG["FAISS_INDEX_PATH"]) / "index.faiss"
    if not index_file.exists() or st.session_state.db_ready:
        return
    try:
        emb = get_embeddings()
        st.session_state.vectorstore = load_vectorstore(CONFIG["FAISS_INDEX_PATH"], emb)
        st.session_state.db_ready = True
        # Rebuild registry from stored PDFs if empty
        if not st.session_state.doc_registry and PDF_STORE.exists():
            from src.loader import load_and_split
            for pdf in PDF_STORE.glob("*.pdf"):
                chunks = load_and_split(str(pdf))
                st.session_state.doc_registry[pdf.name] = {
                    "chunks": len(chunks),
                    "status": "✅ Indexed",
                }
    except Exception:
        pass


_restore_on_startup()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; }
    .app-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .workflow-strip { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.5rem; }
    .wf-step {
        background: #e8f4fd; color: #1a5276; padding: 0.25rem 0.7rem;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600; border: 1px solid #aed6f1;
    }
    .wf-arrow { color: #7f8c8d; line-height: 2; font-size: 0.85rem; }

    .doc-row {
        background: white; border: 1px solid #dee2e6; border-left: 4px solid #2d6a9f;
        border-radius: 8px; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem; font-size: 0.85rem;
    }
    .doc-name  { font-weight: 600; color: #1a3a5c; }
    .doc-meta  { color: #6c757d; font-size: 0.78rem; }

    .score-badge {
        display: inline-block; background: #27ae60; color: white;
        padding: 0.15rem 0.55rem; border-radius: 12px; font-size: 0.78rem; font-weight: 700;
    }
    .chat-user {
        background: #2d6a9f; color: white; padding: 0.7rem 1rem;
        border-radius: 12px 12px 4px 12px; margin: 0.5rem 0;
        max-width: 80%; margin-left: auto; font-size: 0.92rem;
    }
    .chat-answer {
        background: #f0f4f8; color: #1a1a2e; padding: 0.7rem 1rem;
        border-radius: 12px 12px 12px 4px; margin: 0.5rem 0;
        max-width: 80%; font-size: 0.92rem; border-left: 3px solid #27ae60;
    }
    .status-ready   { color: #27ae60; font-weight: 700; }
    .status-pending { color: #e67e22; font-weight: 700; }
    .confirm-box {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
        padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.85rem;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 Document Management")
    st.divider()

    # ── Upload + Ingest ──────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to add to the knowledge base.",
    )

    ingest_btn = st.button(
        "⚡ Ingest & Index",
        disabled=not uploaded_files,
        use_container_width=True,
        type="primary",
    )

    if ingest_btn and uploaded_files:
        emb = get_embeddings()
        for uf in uploaded_files:
            if uf.name in st.session_state.doc_registry:
                st.warning(f"⚠️ `{uf.name}` already indexed — skipping.")
                continue
            with st.spinner(f"Processing {uf.name}…"):
                try:
                    vs, n = ingest_single_pdf(
                        uf, emb,
                        CONFIG["FAISS_INDEX_PATH"],
                        st.session_state.vectorstore,
                        chunk_size=int(CONFIG["CHUNK_SIZE"]),
                        chunk_overlap=int(CONFIG["CHUNK_OVERLAP"]),
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.db_ready    = True
                    st.session_state.doc_registry[uf.name] = {
                        "chunks": n,
                        "status": "✅ Indexed",
                    }
                    st.success(f"✅ `{uf.name}` — {n} chunks indexed")
                except Exception as e:
                    st.error(f"❌ `{uf.name}`: {e}")
                    st.session_state.doc_registry[uf.name] = {
                        "chunks": 0,
                        "status": "❌ Failed",
                    }

    st.divider()

    # ── Document list ────────────────────────────────────────────────────────
    st.markdown("### 📋 Indexed Documents")

    registry = st.session_state.doc_registry

    if not registry:
        st.caption("No documents indexed yet.")
    else:
        # Summary metrics
        mc1, mc2 = st.columns(2)
        mc1.metric("📄 Documents", total_docs())
        mc2.metric("🧩 Chunks", total_chunks())
        st.markdown("")

        with st.expander(f"View all {total_docs()} document(s)", expanded=True):
            for fname, meta in list(registry.items()):
                st.markdown(f"""
                <div class="doc-row">
                    <div class="doc-name">📄 {fname}</div>
                    <div class="doc-meta">{meta['status']} &nbsp;·&nbsp; {meta['chunks']} chunks</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Remove single doc ────────────────────────────────────────
                if st.session_state.confirm_remove == fname:
                    st.markdown(
                        '<div class="confirm-box">⚠️ Remove this document and rebuild the index?</div>',
                        unsafe_allow_html=True,
                    )
                    cc1, cc2 = st.columns(2)
                    if cc1.button("✅ Yes, remove", key=f"yes_{fname}", use_container_width=True):
                        with st.spinner(f"Removing {fname}…"):
                            try:
                                emb = get_embeddings()
                                new_vs = remove_pdf(
                                    fname,
                                    st.session_state.doc_registry,
                                    emb,
                                    CONFIG["FAISS_INDEX_PATH"],
                                )
                                st.session_state.vectorstore = new_vs
                                st.session_state.db_ready    = new_vs is not None
                                st.session_state.confirm_remove = None
                                st.success(f"🗑️ `{fname}` removed.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Removal failed: {e}")
                    if cc2.button("❌ Cancel", key=f"no_{fname}", use_container_width=True):
                        st.session_state.confirm_remove = None
                        st.rerun()
                else:
                    if st.button(
                        f"🗑️ Remove",
                        key=f"remove_{fname}",
                        use_container_width=True,
                    ):
                        st.session_state.confirm_remove = fname
                        st.rerun()

        st.markdown("")

        # ── Remove All ───────────────────────────────────────────────────────
        if st.session_state.confirm_remove_all:
            st.markdown(
                '<div class="confirm-box">⚠️ This will delete ALL documents and reset the FAISS index.</div>',
                unsafe_allow_html=True,
            )
            ra1, ra2 = st.columns(2)
            if ra1.button("✅ Yes, clear all", use_container_width=True):
                with st.spinner("Clearing everything…"):
                    try:
                        remove_all_pdfs(CONFIG["FAISS_INDEX_PATH"])
                        st.session_state.vectorstore        = None
                        st.session_state.db_ready           = False
                        st.session_state.doc_registry       = {}
                        st.session_state.confirm_remove_all = False
                        st.success("🗑️ All documents and index cleared.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Clear failed: {e}")
            if ra2.button("❌ Cancel", key="cancel_all", use_container_width=True):
                st.session_state.confirm_remove_all = False
                st.rerun()
        else:
            if st.button("🗑️ Remove All Documents", use_container_width=True):
                st.session_state.confirm_remove_all = True
                st.rerun()

    st.divider()

    # ── Vector DB status ─────────────────────────────────────────────────────
    st.markdown("### 🗄️ Vector DB Status")
    if st.session_state.db_ready:
        st.markdown('<p class="status-ready">● Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-pending">● Not ready</p>', unsafe_allow_html=True)

    st.divider()

    # ── Config snapshot ───────────────────────────────────────────────────────
    with st.expander("⚙️ Active Configuration"):
        st.json({
            "embedding_model":     CONFIG["EMBEDDING_MODEL"],
            "cross_encoder_model": CONFIG["CROSS_ENCODER_MODEL"],
            "top_k":               CONFIG["TOP_K"],
            "top_n":               CONFIG["TOP_N"],
            "llm_provider":        CONFIG["LLM_PROVIDER"],
            "llm_model":           CONFIG["LLM_MODEL"],
        })

    st.divider()

    # ── Memory stats ──────────────────────────────────────────────────────────
    st.markdown("### 🧠 Conversational Memory")
    msg_count = len(st.session_state.chat_history)
    st.metric("Messages in memory", msg_count)
    if msg_count:
        st.caption(f"Last {min(msg_count, 6)} turns used for query rewriting.")
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main — header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🔍 RAG with Cross-Encoder Reranking</h1>
    <p>Upload PDFs · Build a FAISS index · Ask questions · Get accurate, reranked answers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow-strip">
  <span class="wf-step">📄 PDF Upload</span><span class="wf-arrow">→</span>
  <span class="wf-step">✂️ Chunking</span><span class="wf-arrow">→</span>
  <span class="wf-step">🧮 Embeddings</span><span class="wf-arrow">→</span>
  <span class="wf-step">🗄️ FAISS Retrieval</span><span class="wf-arrow">→</span>
  <span class="wf-step">📋 Top-K Docs</span><span class="wf-arrow">→</span>
  <span class="wf-step">🔁 Cross-Encoder Reranking</span><span class="wf-arrow">→</span>
  <span class="wf-step">📌 Top-N Docs</span><span class="wf-arrow">→</span>
  <span class="wf-step">🤖 LLM Response</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">❓ {entry["query"]}</div>', unsafe_allow_html=True)
        # Show rewritten query if it differs from the original
        if entry.get("standalone_query") and entry["standalone_query"] != entry["query"]:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#6c757d;text-align:right;'
                f'margin-bottom:0.2rem">🔄 Rewritten: <em>{entry["standalone_query"]}</em></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="chat-answer">🤖 {entry["answer"]}</div>', unsafe_allow_html=True)
    st.divider()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
st.markdown("### ❓ Ask a Question")

with st.form("query_form", clear_on_submit=True):
    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What is the main contribution of this paper?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🚀 Submit", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if submitted and query:
    if not st.session_state.db_ready or st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and ingest PDFs first using the sidebar.")
    elif not CONFIG["LLM_API_KEY"]:
        st.error("❌ LLM_API_KEY is missing. Set it in your .env file.")
    else:
        with st.spinner("Running RAG pipeline…"):
            try:
                pipeline = RAGPipeline(CONFIG, vectorstore=st.session_state.vectorstore)
                result   = pipeline.run(query, chat_history=st.session_state.chat_history)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        top_k = int(CONFIG["TOP_K"])
        top_n = int(CONFIG["TOP_N"])

        st.markdown("---")
        standalone = result["standalone_query"]
        query_display = (
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:0.8rem 1.2rem;margin-bottom:0.4rem"><b>🔎 Original Query:</b> {query}</div>'
        )
        if standalone != query:
            query_display += (
                f'<div style="background:#eaf4fb;border:1px solid #aed6f1;border-radius:10px;'
                f'padding:0.6rem 1.2rem;margin-bottom:1rem;font-size:0.88rem">'
                f'🔄 <b>Rewritten Query:</b> <em>{standalone}</em></div>'
            )
        st.markdown(query_display, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"**📄 Top-{top_k} Retrieved Documents** "
                f"<small style='color:#888'>(vector similarity)</small>",
                unsafe_allow_html=True,
            )
            for i, doc in enumerate(result["retrieved_docs"], 1):
                source = doc.metadata.get("source", "unknown")
                page   = doc.metadata.get("page", "?")
                with st.expander(f"[{i}] {Path(source).name} · page {page}"):
                    st.caption(f"`{source}` | Page {page}")
                    st.write(doc.page_content)

        with col2:
            st.markdown(
                f"**🔁 Top-{top_n} Reranked Documents** "
                f"<small style='color:#888'>(Cross-Encoder scores)</small>",
                unsafe_allow_html=True,
            )
            for i, (doc, score) in enumerate(result["reranked_docs"], 1):
                source = doc.metadata.get("source", "unknown")
                page   = doc.metadata.get("page", "?")
                with st.expander(f"[{i}] {Path(source).name} · page {page} · score {score:.4f}"):
                    st.caption(f"`{source}` | Page {page}")
                    st.markdown(
                        f'<span class="score-badge">Score: {score:.4f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(doc.page_content)

        st.markdown("### 🤖 Final Answer")
        st.markdown(
            f'<div class="chat-answer" style="max-width:100%">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        st.session_state.chat_history.append({
            "query":           query,
            "standalone_query": result["standalone_query"],
            "answer":          result["answer"],
        })

elif submitted and not query:
    st.warning("Please enter a question before submitting.")
