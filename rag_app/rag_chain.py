"""
RAG chain – full retrieval-augmented generation with Ollama (Llama3).

Ties together:
  • FAISS-based retrieval  (mxbai-embed-large-v1)
  • LLM answer generation  (Ollama / llama3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    TOP_K,
    VECTOR_STORE_DIR,
)
from rag_app.retriever import get_retriever

# ── Prompt ───────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "context provided below. If the context does not contain enough information, "
    "say so clearly – do not make things up.\n\n"
    "--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Container for a RAG query result."""
    query: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [f"\n{'='*70}", f"Query: {self.query}", f"{'='*70}"]
        lines.append(f"\n📝 Answer:\n{self.answer}")
        if self.source_documents:
            lines.append(f"\n{'─'*70}")
            lines.append("📚 Sources:")
            for i, doc in enumerate(self.source_documents, 1):
                source = doc.metadata.get("source", "unknown")
                lines.append(f"  [{i}] {source}")
        lines.append(f"\n{'='*70}\n")
        return "\n".join(lines)


# ── RAG Engine ───────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Full Retrieval-Augmented Generation engine.

    Pipeline:
        question → retriever → format context → LLM → answer
    """

    def __init__(
        self,
        store_dir: Path = VECTOR_STORE_DIR,
        top_k: int = TOP_K,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = OLLAMA_TEMPERATURE,
    ) -> None:
        self.retriever = get_retriever(store_dir=store_dir, top_k=top_k)
        self.llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self.chain = (
            {
                "context": self.retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline: retrieve → generate."""
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return RAGResult(query=question, answer=answer, source_documents=docs)

    def retrieve_only(self, question: str) -> List[Document]:
        """Return raw retrieved chunks without LLM generation."""
        return self.retriever.invoke(question)
