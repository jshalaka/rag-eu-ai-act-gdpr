"""BM25 keyword index with optional JSON-lines persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_eu_law.domain.exceptions import IndexingError
from rag_eu_law.domain.models import Chunk, ScoredChunk, SearchFilter
from rag_eu_law.indexing.text import embedding_text, tokenize


class Bm25KeywordIndex:
    """`KeywordIndex` using Okapi BM25 over heading + text.

    The heading is indexed so that "Article 22" and "Recital 47" queries hit by number.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._chunks: dict[str, Chunk] = {}
        self._bm25: Any = None
        self._order: list[str] = []
        if path is not None and path.is_file():
            self._load(path)

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._bm25 = None
        if self._path is not None:
            self._save(self._path)

    def search(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if not tokens or not self._chunks:
            return []
        bm25 = self._index()
        scores = bm25.get_scores(tokens)
        ranked = sorted(zip(self._order, scores, strict=True), key=lambda p: (-p[1], p[0]))
        results = [
            ScoredChunk(chunk=self._chunks[chunk_id], score=float(score))
            for chunk_id, score in ranked
            if score > 0
            and (search_filter is None or search_filter.matches(self._chunks[chunk_id]))
        ]
        return results[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def _index(self) -> Any:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415 - optional 'index' extra, imported lazily

            self._order = sorted(self._chunks)
            corpus = [tokenize(embedding_text(self._chunks[i])) for i in self._order]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [self._chunks[i].model_dump_json() for i in sorted(self._chunks)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            self._chunks = {c.chunk_id: c for c in map(Chunk.model_validate_json, lines) if c}
        except (OSError, ValueError) as error:
            raise IndexingError(f"Cannot read keyword index {path}: {error}") from error
