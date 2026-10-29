"""Indexing pipeline: embed chunks, then store them in both indexes."""

from __future__ import annotations

import logging

from rag_eu_law.domain.constants import EMBEDDING_BATCH_SIZE
from rag_eu_law.domain.models import Chunk
from rag_eu_law.domain.protocols import Embedder, KeywordIndex, VectorStore
from rag_eu_law.indexing.text import embedding_text

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Idempotent: both indexes upsert by `chunk_id`."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._batch_size = batch_size

    def index(self, chunks: list[Chunk]) -> int:
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            embeddings = self._embedder.embed([embedding_text(chunk) for chunk in batch])
            self._vector_store.upsert(batch, embeddings)
        self._keyword_index.upsert(chunks)
        logger.info("indexed %d chunks", len(chunks))
        return len(chunks)
