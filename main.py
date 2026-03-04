"""
main.py – CLI entry point for the RAG application.

Usage
─────
  1. Place your .txt / .md / .pdf files in the  documents/  folder.
  2. Ingest:       python main.py ingest
  3. Query:        python main.py query "What is …?"
  4. Interactive:  python main.py interactive
  5. Streamlit UI: streamlit run streamlit_app.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag_app.config import DOCUMENTS_DIR, VECTOR_STORE_DIR
from rag_app.ingest import build_vector_store
from rag_app.rag_chain import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── CLI commands ─────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents and build the FAISS index."""
    logger.info("Starting ingestion from %s …", DOCUMENTS_DIR)
    build_vector_store()
    logger.info("Ingestion complete. Index saved to %s", VECTOR_STORE_DIR)


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single query against the vector store."""
    engine = RAGEngine()
    result = engine.query(args.question)
    print(result.pretty())


def cmd_interactive(args: argparse.Namespace) -> None:
    """Start an interactive query session."""
    engine = RAGEngine()
    print("\n💡 RAG Interactive Mode  (type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if not question:
            continue
        result = engine.query(question)
        print(result.pretty())


# ── Argument parser ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI – LangChain + FAISS + mxbai-embed-large-v1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    subparsers.add_parser("ingest", help="Ingest documents into FAISS")

    # query
    q_parser = subparsers.add_parser("query", help="Query the vector store")
    q_parser.add_argument("question", type=str, help="Your question")

    # interactive
    subparsers.add_parser("interactive", help="Interactive query shell")

    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "interactive": cmd_interactive,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
