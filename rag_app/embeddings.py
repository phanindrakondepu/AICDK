"""
Custom LangChain-compatible embedding wrapper for the
mixedbread-ai/mxbai-embed-large-v1 model via sentence-transformers.

The model requires a special query prefix for retrieval tasks while
document embeddings are generated without any prefix.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from rag_app.config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, QUERY_PREFIX


class MxbaiEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around mxbai-embed-large-v1."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
        query_prefix: str = QUERY_PREFIX,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix

    # ── Documents (no prefix) ────────────────────────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks (no query prefix)."""
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.tolist()

    # ── Queries (with retrieval prefix) ──────────────────────────────────
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the retrieval prefix."""
        prefixed = f"{self.query_prefix}{text}"
        embedding = self.model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()
