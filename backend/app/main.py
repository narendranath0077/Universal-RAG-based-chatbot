from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import EXTRACT_DIR, UPLOAD_DIR
from .document_loader import SUPPORTED_EXTENSIONS, extract_if_zip, load_documents
from .rag_service import RAGService
from .schemas import AskRequest, AskResponse, IngestResponse

app = FastAPI(title="Universal Agentic RAG API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()


class PathIngestRequest(BaseModel):
    file_paths: list[str] = Field(default_factory=list)


def _collect_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS.union({".zip"}) else []
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS.union({".zip"})]
    return []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: list[UploadFile] = File(...)):
    all_docs = []
    ingested = 0

    for file in files:
        safe_name = Path(file.filename or "uploaded_file").name
        saved_path = UPLOAD_DIR / safe_name

        with saved_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        resolved_files = extract_if_zip(saved_path, EXTRACT_DIR)
        for path in resolved_files:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            all_docs.extend(load_documents(path))
            ingested += 1

    if not all_docs:
        raise HTTPException(status_code=400, detail="No supported files found for ingestion.")

    chunks = rag_service.add_documents(all_docs)
    return IngestResponse(
        ingested_files=ingested,
        chunks_added=chunks,
        message="Files ingested and indexed successfully.",
    )


@app.post("/ingest/paths", response_model=IngestResponse)
def ingest_paths(payload: PathIngestRequest):
    all_docs = []
    ingested = 0

    for raw in payload.file_paths:
        base_path = Path(raw).expanduser().resolve()
        for path in _collect_supported_files(base_path):
            resolved_files = extract_if_zip(path, EXTRACT_DIR)
            for each in resolved_files:
                if each.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                all_docs.extend(load_documents(each))
                ingested += 1

    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid files found in given paths.")

    chunks = rag_service.add_documents(all_docs)
    return IngestResponse(
        ingested_files=ingested,
        chunks_added=chunks,
        message="Paths ingested and indexed successfully.",
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    result = rag_service.ask(payload.query)
    return AskResponse(**result)
