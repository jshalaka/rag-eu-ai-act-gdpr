"""Embedder implementations."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from rag_eu_law.domain.constants import (
    BGE_QUERY_PREFIX,
    EMBEDDING_BATCH_SIZE,
    HASHING_EMBEDDER_DIMENSION,
)
from rag_eu_law.domain.exceptions import IndexingError
from rag_eu_law.indexing.text import tokenize


class SentenceTransformerEmbedder:
    """Local dense embeddings. The model is loaded on first use."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load().encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:  # backend errors are varied; surface one type
            raise IndexingError(f"Embedding failed with {self._model_name}: {error}") from error
        return [[float(value) for value in vector] for vector in vectors]

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional 'index' extra, imported lazily

            self._model = SentenceTransformer(self._model_name)
        return self._model


class FastEmbedEmbedder:
    """ONNX embeddings (no PyTorch): same bge model, small install, CPU only.

    Vectors match `SentenceTransformerEmbedder` for the same model (cosine 1.0 on a test
    sentence), so an index built with one can be queried with the other.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return int(self._load().embedding_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load().embed(texts, batch_size=self._batch_size)
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as error:  # ONNX runtime and download errors are varied
            raise IndexingError(f"Embedding failed with {self._model_name}: {error}") from error

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # noqa: PLC0415 - optional 'index-lite' extra

            self._model = TextEmbedding(self._model_name, cache_dir=self._cache_dir)
        return self._model


class HashingEmbedder:
    """Dependency-free bag-of-words embedder (feature hashing). Fast and deterministic.

    Not semantic: use it for tests and offline smoke runs, not for answer quality.
    """

    def __init__(self, dimension: int = HASHING_EMBEDDER_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
