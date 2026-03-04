# 🧠 RAG Application — LangChain + FAISS + Ollama Llama 3

A full-stack Retrieval-Augmented Generation system with a Streamlit UI.

| Component | Technology |
|-----------|-----------|
| **Framework** | LangChain |
| **Vector Store** | FAISS (CPU) |
| **Embedding Model** | [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) |
| **LLM** | Ollama / Llama 3 |
| **UI** | Streamlit |

## 📁 Project Structure

```
AICDK/
├── main.py                  # CLI entry point
├── streamlit_app.py         # Streamlit UI (Query + Ingest tabs)
├── requirements.txt         # Python dependencies
├── documents/               # Drop your .txt / .md / .pdf files here
│   └── sample.txt           # Example document
├── vector_store/            # FAISS index (auto-created)
└── rag_app/
    ├── __init__.py
    ├── config.py            # All tuneable settings
    ├── embeddings.py        # LangChain-compatible mxbai wrapper
    ├── ingest.py            # Document loading → chunking → indexing
    ├── retriever.py         # LangChain retriever over FAISS
    └── rag_chain.py         # Full RAG engine (retrieval + Ollama LLM)
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure Ollama is running with Llama 3

```bash
ollama run llama3
```

### 3. Launch the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### 4. Or use the CLI

```bash
# Ingest documents
python main.py ingest

# Single question
python main.py query "What is RAG?"

# Interactive mode
python main.py interactive
```

## 🖥️ Streamlit UI

The UI has two tabs:

- **🔍 Query** — Chat interface to ask questions. The system retrieves relevant
  chunks from FAISS, sends them as context to Llama 3 via Ollama, and displays
  the generated answer with expandable source references.

- **📥 Ingest** — Upload `.txt`, `.md`, or `.pdf` files directly through the
  browser. Documents are chunked, embedded, and merged into the FAISS index.
  You can also rebuild the index from the `documents/` folder or delete it.

## ⚙️ Configuration

Edit `rag_app/config.py` to change:

- `EMBEDDING_MODEL_NAME` — HuggingFace model ID
- `EMBEDDING_DEVICE` — `"cpu"` or `"cuda"`
- `OLLAMA_MODEL` — Ollama model name (default: `llama3`)
- `OLLAMA_BASE_URL` — Ollama server URL
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text splitting parameters
- `TOP_K` — number of retrieved chunks per query
"# AICDK" 
