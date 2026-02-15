from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class IngestResponse(BaseModel):
    ingested_files: int
    chunks_added: int
    message: str
