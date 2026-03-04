"""
Document ingestion pipeline.

Reads files from the `documents/` directory (or uploaded files),
splits them into chunks, embeds them with mxbai-embed-large-v1,
and persists a FAISS index to disk.

Supported file types: .txt, .md, .pdf
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import IO, List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
    VECTOR_STORE_DIR,
)
from rag_app.embeddings import MxbaiEmbeddings

logger = logging.getLogger(__name__)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_documents(docs_dir: Path) -> List[Document]:
    """Load .txt, .md, and .pdf files from the given directory."""
    documents: List[Document] = []

    # Plain text files
    txt_loader = DirectoryLoader(
        str(docs_dir), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=True,
    )
    documents.extend(txt_loader.load())

    # Markdown files
    md_loader = DirectoryLoader(
        str(docs_dir), glob="**/*.md", loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    documents.extend(md_loader.load())

    # PDF files
    pdf_loader = DirectoryLoader(
        str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents.extend(pdf_loader.load())

    logger.info("Loaded %d document(s) from %s", len(documents), docs_dir)
    return documents


# ── Splitting ────────────────────────────────────────────────────────────────

def _split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


# ── FAISS Indexing ───────────────────────────────────────────────────────────

def build_vector_store(
    docs_dir: Path = DOCUMENTS_DIR,
    store_dir: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """
    End-to-end ingestion:
      1. Load raw documents
      2. Chunk them
      3. Embed & index with FAISS
      4. Persist index to disk

    Returns the FAISS vector store instance.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    documents = _load_documents(docs_dir)
    if not documents:
        raise FileNotFoundError(
            f"No documents found in {docs_dir}. "
            "Add .txt or .md files and re-run ingestion."
        )

    chunks = _split_documents(documents)

    logger.info("Creating embeddings and building FAISS index …")
    embedding_model = MxbaiEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding_model)

    vector_store.save_local(str(store_dir))
    logger.info("FAISS index saved to %s", store_dir)

    return vector_store


# ── Loading an existing index ────────────────────────────────────────────────

def load_vector_store(store_dir: Path = VECTOR_STORE_DIR) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embedding_model = MxbaiEmbeddings()
    vector_store = FAISS.load_local(
        str(store_dir), embedding_model, allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded from %s", store_dir)
    return vector_store


# ── Helpers for uploaded files (Streamlit) ───────────────────────────────────

def _load_single_file(file_path: Path, filename: str) -> List[Document]:
    """Load a single file based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(str(file_path))
    else:  # .txt and fallback
        loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def ingest_uploaded_files(
    uploaded_files: list,
    store_dir: Path = VECTOR_STORE_DIR,
) -> int:
    """
    Ingest files uploaded through Streamlit.

    Writes each uploaded file to a temp location, loads it, chunks it,
    and either creates a new FAISS index or merges into the existing one.

    Returns the total number of chunks indexed.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Document] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        try:
            docs = _load_single_file(tmp_path, uploaded.name)
            # Tag each doc with the original filename
            for doc in docs:
                doc.metadata["source"] = uploaded.name
            chunks = _split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            tmp_path.unlink(missing_ok=True)

    if not all_chunks:
        return 0

    embedding_model = MxbaiEmbeddings()

    # Merge into existing index or create a new one
    if (store_dir / "index.faiss").exists():
        existing = FAISS.load_local(
            str(store_dir), embedding_model, allow_dangerous_deserialization=True
        )
        new_store = FAISS.from_documents(all_chunks, embedding_model)
        existing.merge_from(new_store)
        existing.save_local(str(store_dir))
        logger.info("Merged %d chunk(s) into existing index", len(all_chunks))
    else:
        vector_store = FAISS.from_documents(all_chunks, embedding_model)
        vector_store.save_local(str(store_dir))
        logger.info("Created new index with %d chunk(s)", len(all_chunks))

    return len(all_chunks)


def get_index_stats(store_dir: Path = VECTOR_STORE_DIR) -> dict:
    """Return basic stats about the persisted FAISS index."""
    index_path = store_dir / "index.faiss"
    if not index_path.exists():
        return {"exists": False, "num_vectors": 0, "size_mb": 0.0}

    embedding_model = MxbaiEmbeddings()
    vs = FAISS.load_local(
        str(store_dir), embedding_model, allow_dangerous_deserialization=True
    )
    num_vectors = vs.index.ntotal
    size_mb = index_path.stat().st_size / (1024 * 1024)
    return {"exists": True, "num_vectors": num_vectors, "size_mb": round(size_mb, 2)}
