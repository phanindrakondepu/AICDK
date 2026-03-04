"""
Retriever module.

Wraps the FAISS vector store in a LangChain retriever interface so it
can plug directly into chains / agents.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from rag_app.config import TOP_K, VECTOR_STORE_DIR
from rag_app.ingest import load_vector_store


def get_retriever(
    store_dir: Path = VECTOR_STORE_DIR,
    top_k: int = TOP_K,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever backed by the persisted FAISS index.

    Args:
        store_dir: Path where the FAISS index is stored.
        top_k: Number of chunks to retrieve per query.

    Returns:
        A VectorStoreRetriever instance.
    """
    vector_store = load_vector_store(store_dir)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
