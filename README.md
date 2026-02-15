# Universal-RAG-based-chatbot

An **agentic universal QA RAG system** with:

- Multi-format ingestion (file upload, zip upload, direct file paths, and directories)
- Chunking + metadata-aware embeddings
- FAISS vector database
- Hybrid reranking (BM25 + CrossEncoder reranker)
- Local LLM + embeddings through **Ollama**
- LangChain + LangGraph orchestration
- Streamlit chatbot UI (royal-black + glass-smoke look)

## Architecture

- **Backend API (FastAPI)**
  - `/ingest/upload` → upload files or zip archive
  - `/ingest/paths` → ingest local file paths and folders
  - `/ask` → ask question against indexed corpus
- **RAG Pipeline**
  1. Parse files
  2. Chunk with overlap
  3. Store metadata per chunk
  4. Embed + store in FAISS
  5. Retrieve top-k
  6. Rerank with BM25 and CrossEncoder
  7. Hybrid score merge
  8. Answer generation with Ollama chat model
- **Frontend (Streamlit)**
  - Chatbot interface
  - Sidebar ingestion tools
  - Source list and hybrid scores

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

```bash
./scripts/run_backend.sh
./scripts/run_frontend.sh
```

Open Streamlit at `http://localhost:8501`.

## Notes

- Supported extensions: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.xlsx`, `.json`, `.zip`.
- ZIP extraction is path-safe (prevents path traversal payloads).
- FAISS index persists at `backend/data/faiss_index/main`.
- If the CrossEncoder model cannot load (offline constraints), reranking gracefully falls back to BM25-only behavior.
