"""
Centralized configuration for the RAG application.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DEVICE = "cpu"  # change to "cuda" if GPU is available

# Query prefix required by the mxbai model for retrieval tasks
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.3

# ── Text Splitting ───────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 4  # number of chunks returned per query

# ── Supported file extensions ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
