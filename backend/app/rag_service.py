from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANKER_MODEL,
    FINAL_K,
    INDEX_DIR,
    TOP_K,
)


class AgentState(TypedDict):
    query: str
    retrieved: list[Document]
    reranked: list[Document]
    answer: str


@dataclass
class RAGService:
    vectorstore: FAISS | None = None
    cross_encoder: CrossEncoder | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
        self.llm = ChatOllama(model=DEFAULT_CHAT_MODEL, temperature=0.1)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a universal enterprise QA assistant.
Answer strictly using context. If information is missing, say so clearly.

Question: {question}

Context:
{context}

Give a concise and accurate answer.
""".strip()
        )
        self._load_index_if_present()
        self.graph = self._build_graph()

    def _ensure_cross_encoder(self) -> bool:
        if self.cross_encoder is not None:
            return True
        try:
            self.cross_encoder = CrossEncoder(DEFAULT_RERANKER_MODEL)
            return True
        except Exception:
            return False

    def _index_path(self) -> Path:
        return INDEX_DIR / "main"

    def _load_index_if_present(self) -> None:
        index_path = self._index_path()
        if index_path.exists():
            self.vectorstore = FAISS.load_local(
                str(index_path), self.embeddings, allow_dangerous_deserialization=True
            )

    def _save_index(self) -> None:
        if self.vectorstore is not None:
            self.vectorstore.save_local(str(self._index_path()))

    def add_documents(self, docs: list[Document]) -> int:
        chunks = self.splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid4())
            chunk.metadata["char_count"] = len(chunk.page_content)

        if not chunks:
            return 0

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        self._save_index()
        return len(chunks)

    def _retrieve(self, state: AgentState) -> AgentState:
        if self.vectorstore is None:
            return {**state, "retrieved": []}
        docs = self.vectorstore.similarity_search(state["query"], k=TOP_K)
        return {**state, "retrieved": docs}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _hybrid_rerank(self, state: AgentState) -> AgentState:
        docs = state.get("retrieved", [])
        if not docs:
            return {**state, "reranked": []}

        corpus = [d.page_content for d in docs]
        tokenized = [self._tokenize(x) for x in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = np.array(bm25.get_scores(self._tokenize(state["query"])), dtype=float)

        if self._ensure_cross_encoder():
            pairs = [(state["query"], d.page_content) for d in docs]
            ce_scores = np.array(self.cross_encoder.predict(pairs), dtype=float)
            ce_norm = (ce_scores - np.min(ce_scores)) / ((np.max(ce_scores) - np.min(ce_scores)) + 1e-9)
        else:
            ce_scores = np.zeros(len(docs), dtype=float)
            ce_norm = np.zeros(len(docs), dtype=float)

        bm25_norm = (bm25_scores - np.min(bm25_scores)) / ((np.max(bm25_scores) - np.min(bm25_scores)) + 1e-9)
        hybrid = 0.4 * bm25_norm + 0.6 * ce_norm if len(docs) > 1 else bm25_norm

        ranked_idx = list(np.argsort(-hybrid))[: min(FINAL_K, len(docs))]

        reranked: list[Document] = []
        for idx in ranked_idx:
            doc = docs[int(idx)]
            doc.metadata["bm25_score"] = float(bm25_scores[int(idx)])
            doc.metadata["cross_encoder_score"] = float(ce_scores[int(idx)])
            doc.metadata["hybrid_score"] = float(hybrid[int(idx)])
            reranked.append(doc)

        return {**state, "reranked": reranked}

    def _generate(self, state: AgentState) -> AgentState:
        docs = state.get("reranked", [])
        if not docs:
            return {**state, "answer": "I don't have indexed context yet. Please upload documents first."}

        context = "\n\n".join(
            [f"[Source: {d.metadata.get('file_name', 'unknown')}]\n{d.page_content}" for d in docs]
        )
        chain = self.prompt | self.llm
        result = chain.invoke({"question": state["query"], "context": context})
        return {**state, "answer": result.content}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._hybrid_rerank)
        workflow.add_node("generate", self._generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask(self, query: str) -> dict:
        result = self.graph.invoke({"query": query, "retrieved": [], "reranked": [], "answer": ""})
        docs = result.get("reranked", [])
        sources = [
            {
                "file_name": d.metadata.get("file_name", "unknown"),
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id"),
                "hybrid_score": d.metadata.get("hybrid_score"),
            }
            for d in docs
        ]
        return {"answer": result.get("answer", ""), "sources": sources}
