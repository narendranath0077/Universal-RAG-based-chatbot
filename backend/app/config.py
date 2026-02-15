from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "faiss_index"
EXTRACT_DIR = DATA_DIR / "extracted"

for path in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, EXTRACT_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "llama3.1:8b"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8
FINAL_K = 4
