"""
streamlit_app.py – Streamlit UI for the RAG application.

Two tabs:
  🔍 Query   – Ask questions, get LLM-generated answers with sources
  📥 Ingest  – Upload documents, build / update the FAISS index
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st

from rag_app.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    VECTOR_STORE_DIR,
)
from rag_app.ingest import (
    build_vector_store,
    get_index_stats,
    ingest_uploaded_files,
)
from rag_app.rag_chain import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/3d-fluency/94/brain.png",
        width=60,
    )
    st.markdown("### ⚙️ Configuration")
    st.divider()

    st.markdown(f"**Embedding Model**")
    st.code(EMBEDDING_MODEL_NAME, language=None)

    st.markdown(f"**LLM**")
    st.code(f"Ollama / {OLLAMA_MODEL}", language=None)

    st.markdown(f"**Top‑K Chunks**")
    st.code(str(TOP_K), language=None)

    st.divider()

    # Index stats
    stats = get_index_stats()
    if stats["exists"]:
        st.success("✅ FAISS index loaded")
        col1, col2 = st.columns(2)
        col1.metric("Vectors", stats["num_vectors"])
        col2.metric("Size", f"{stats['size_mb']} MB")
    else:
        st.warning("⚠️ No FAISS index found. Ingest documents first.")

    st.divider()
    st.caption("Built with LangChain · FAISS · Ollama")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 RAG System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">LangChain &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; '
    'mxbai-embed-large-v1 &nbsp;·&nbsp; Ollama Llama 3</p>',
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_query, tab_ingest = st.tabs(["🔍 Query", "📥 Ingest"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 — QUERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_query:
    st.markdown("### Ask a question about your documents")

    # Check if index exists
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        st.info(
            "📭 No vector index found. Go to the **📥 Ingest** tab to upload "
            "and index your documents first."
        )
    else:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 View Sources", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f'<div class="source-box">'
                                f"<strong>Source {i}:</strong> {src['name']}<br>"
                                f"<em>{src['preview']}</em></div>",
                                unsafe_allow_html=True,
                            )

        # Chat input
        if question := st.chat_input("Type your question here…"):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        engine = RAGEngine()
                        result = engine.query(question)

                        st.markdown(result.answer)

                        sources = []
                        if result.source_documents:
                            with st.expander("📚 View Sources", expanded=False):
                                for i, doc in enumerate(result.source_documents, 1):
                                    name = doc.metadata.get("source", "unknown")
                                    preview = doc.page_content[:200] + "…"
                                    sources.append({"name": name, "preview": preview})
                                    st.markdown(
                                        f'<div class="source-box">'
                                        f"<strong>Source {i}:</strong> {name}<br>"
                                        f"<em>{preview}</em></div>",
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result.answer,
                                "sources": sources,
                            }
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2 — INGEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ingest:
    st.markdown("### Upload & Index Documents")
    st.markdown(
        "Upload `.txt`, `.md`, or `.pdf` files. They will be chunked, embedded "
        "with **mxbai-embed-large-v1**, and stored in the FAISS vector index."
    )

    col_upload, col_status = st.columns([2, 1])

    with col_upload:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Supported formats: .txt, .md, .pdf",
        )

        if uploaded_files:
            st.markdown("**Uploaded files:**")
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                st.markdown(f"- 📄 `{f.name}` ({size_kb:.1f} KB)")

            if st.button("🚀 Ingest Uploaded Files", type="primary", use_container_width=True):
                with st.spinner("Embedding and indexing documents…"):
                    start = time.time()
                    try:
                        num_chunks = ingest_uploaded_files(uploaded_files)
                        elapsed = time.time() - start
                        st.success(
                            f"✅ Indexed **{num_chunks} chunks** from "
                            f"**{len(uploaded_files)} file(s)** in {elapsed:.1f}s"
                        )
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")

    with col_status:
        st.markdown("**Index Status**")
        stats = get_index_stats()
        if stats["exists"]:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>{stats["num_vectors"]}</h3>'
                f"<p>vectors indexed</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>{stats["size_mb"]} MB</h3>'
                f"<p>index size on disk</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No index yet.")

    st.divider()

    # ── Ingest from documents/ folder ────────────────────────────────────
    st.markdown("### Or: Ingest from `documents/` folder")
    st.markdown(
        f"Place files in `{DOCUMENTS_DIR}` and click the button below "
        "to rebuild the entire index from that folder."
    )

    col_folder, col_danger = st.columns([1, 1])

    with col_folder:
        if st.button("📂 Ingest from documents/ folder", use_container_width=True):
            with st.spinner("Building index from documents/ folder…"):
                start = time.time()
                try:
                    build_vector_store()
                    elapsed = time.time() - start
                    st.success(f"✅ Index rebuilt in {elapsed:.1f}s")
                    time.sleep(1)
                    st.rerun()
                except FileNotFoundError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col_danger:
        if st.button("🗑️ Delete Index", use_container_width=True):
            if VECTOR_STORE_DIR.exists():
                shutil.rmtree(VECTOR_STORE_DIR)
                st.warning("Index deleted.")
                time.sleep(1)
                st.rerun()
            else:
                st.info("No index to delete.")
