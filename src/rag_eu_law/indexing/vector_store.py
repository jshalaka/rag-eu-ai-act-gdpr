"""VectorStore implementations: Chroma (persistent, default) and in-memory (tests)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rag_eu_law.domain.constants import CHROMA_COLLECTION
from rag_eu_law.domain.exceptions import IndexingError, RetrievalError
from rag_eu_law.domain.models import Chunk, ScoredChunk, SearchFilter

CHUNK_JSON_KEY = "chunk_json"


class InMemoryVectorStore:
    """Exact cosine search over a dict. Fine for tests and tiny corpora."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise IndexingError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items[chunk.chunk_id] = (chunk, embedding)

    def query(
        self, embedding: list[float], top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine(embedding, vector))
            for chunk, vector in self._items.values()
            if search_filter is None or search_filter.matches(chunk)
        ]
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]

    def count(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    """Persistent local vector store backed by Chroma (cosine distance)."""

    def __init__(self, directory: Path, collection_name: str = CHROMA_COLLECTION) -> None:
        import chromadb  # noqa: PLC0415 - optional 'index' extra, imported lazily

        try:
            client = chromadb.PersistentClient(path=str(directory))
            self._collection: Any = client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as error:
            raise IndexingError(f"Cannot open Chroma store at {directory}: {error}") from error

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise IndexingError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[_metadata(chunk) for chunk in chunks],
            )
        except Exception as error:
            raise IndexingError(f"Chroma upsert failed: {error}") from error

    def query(
        self, embedding: list[float], top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=_where(search_filter),
                include=["metadatas", "distances"],
            )
        except Exception as error:
            raise RetrievalError(f"Chroma query failed: {error}") from error
        return [
            ScoredChunk(chunk=Chunk.model_validate_json(meta[CHUNK_JSON_KEY]), score=1.0 - dist)
            for meta, dist in zip(result["metadatas"][0], result["distances"][0], strict=True)
        ]

    def count(self) -> int:
        return int(self._collection.count())


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    """Filterable fields plus the full chunk as JSON (Chroma rejects None values)."""
    fields = {
        "regulation": chunk.regulation.value,
        "article": chunk.article,
        "recital": chunk.recital,
        "annex": chunk.annex,
    }
    present: dict[str, str | int] = {k: v for k, v in fields.items() if v is not None}
    present[CHUNK_JSON_KEY] = chunk.model_dump_json()
    return present


def _where(search_filter: SearchFilter | None) -> dict[str, Any] | None:
    constraints = search_filter.constraints() if search_filter else {}
    if not constraints:
        return None
    clauses = [{field: {"$eq": value}} for field, value in constraints.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
