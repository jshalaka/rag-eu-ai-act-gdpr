"""Rerankers. Off by default; enable with `RAG_RERANK_ENABLED=true`."""

from __future__ import annotations

from typing import Any

from rag_eu_law.domain.exceptions import RetrievalError
from rag_eu_law.domain.models import ScoredChunk
from rag_eu_law.indexing.text import embedding_text


class NoOpReranker:
    """Keeps the incoming order."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a cross-encoder. The model loads on first use."""

    def __init__(self, model_name: str, model: Any = None) -> None:
        self._model_name = model_name
        self._model: Any = model  # injectable for tests; loaded lazily otherwise

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, embedding_text(item.chunk)) for item in candidates]
        try:
            scores = self._load().predict(pairs)
        except Exception as error:
            raise RetrievalError(f"Reranking failed with {self._model_name}: {error}") from error
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].chunk.chunk_id),
        )
        return [ScoredChunk(chunk=item.chunk, score=float(score)) for item, score in ranked[:top_k]]

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415 - optional extra

            self._model = CrossEncoder(self._model_name)
        return self._model
