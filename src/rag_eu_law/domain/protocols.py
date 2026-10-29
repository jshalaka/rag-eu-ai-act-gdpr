"""Narrow interfaces. High-level pipelines depend only on these."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from rag_eu_law.domain.models import (
    Answer,
    Chunk,
    LegalUnit,
    Page,
    Regulation,
    RunRecord,
    ScoredChunk,
    SearchFilter,
)


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[Page]:
        """Return one `Page` per PDF page. Raise `SourceDocumentError` if unreadable."""
        ...


@runtime_checkable
class RegulationDetector(Protocol):
    def detect(self, pages: list[Page]) -> Regulation:
        """Identify which regulation a document contains. Raise `SourceDocumentError` if unknown."""
        ...


@runtime_checkable
class TextCleaner(Protocol):
    def clean(self, pages: list[Page]) -> list[Page]:
        """Strip headers, footers, page numbers and noise; repair hyphenation."""
        ...


@runtime_checkable
class StructureParser(Protocol):
    def parse(self, pages: list[Page], regulation: Regulation) -> list[LegalUnit]:
        """Split cleaned pages into articles, recitals and annexes."""
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        """Turn legal units into retrievable chunks with deterministic ids."""
        ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per document text, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models treat queries differently from documents)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or overwrite by `chunk_id` (idempotent)."""
        ...

    def query(
        self, embedding: list[float], top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to `top_k` chunks, best first."""
        ...

    def count(self) -> int: ...


@runtime_checkable
class KeywordIndex(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or overwrite by `chunk_id` (idempotent)."""
        ...

    def search(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to `top_k` chunks, best first."""
        ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text. Raise `GenerationError` on failure."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    def generate(self, question: str, context: list[ScoredChunk]) -> Answer: ...


@runtime_checkable
class RunStore(Protocol):
    def save(self, record: RunRecord) -> None:
        """Insert a run. Raise `RunStoreError` on failure."""
        ...

    def get(self, run_id: str) -> RunRecord | None: ...

    def list_runs(self, limit: int, user: str | None = None) -> list[RunRecord]:
        """Most recent first, optionally only this user's."""
        ...

    def rate(self, run_id: str, rating: int, note: str | None) -> None:
        """Set the reviewer's rating and note. Raise `RunStoreError` if the run is unknown."""
        ...
