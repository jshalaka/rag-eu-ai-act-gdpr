"""Hybrid retrieval: dense + BM25 + explicit-reference lookups, merged with RRF."""

from __future__ import annotations

from rag_eu_law.domain.constants import (
    DEFAULT_CANDIDATES_PER_RETRIEVER,
    REFERENCE_CANDIDATES,
    REFERENCE_LIST_WEIGHT,
    RERANK_POOL_MULTIPLIER,
)
from rag_eu_law.domain.exceptions import RetrievalError
from rag_eu_law.domain.models import Regulation, ScoredChunk, SearchFilter
from rag_eu_law.domain.protocols import Embedder, KeywordIndex, Reranker, VectorStore
from rag_eu_law.retrieval.fusion import RankedList, reciprocal_rank_fusion
from rag_eu_law.retrieval.references import QueryReferences, parse_references
from rag_eu_law.retrieval.reranker import NoOpReranker


class HybridRetriever:
    """`Retriever` that merges dense, keyword and explicit-reference rankings.

    Legal questions often name an article ("Article 22"). Neither BM25 nor embeddings rank
    such a lookup reliably, so each named article, recital or annex also contributes its own
    filtered ranking with a higher weight.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        reranker: Reranker | None = None,
        candidates: int = DEFAULT_CANDIDATES_PER_RETRIEVER,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._reranker: Reranker = reranker or NoOpReranker()
        self._candidates = candidates

    def retrieve(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        if top_k < 1:
            raise RetrievalError(f"top_k must be at least 1, got {top_k}")
        if self._vector_store.count() == 0:
            raise RetrievalError("The index is empty. Run ingestion first.")
        references = parse_references(query)
        base = _apply_query_regulation(search_filter, references)
        query_vector = self._embedder.embed_query(query)
        ranked = [
            RankedList(self._vector_store.query(query_vector, self._candidates, base)),
            RankedList(self._keyword_index.search(query, self._candidates, base)),
            *self._reference_lists(query_vector, references, base),
        ]
        pool = reciprocal_rank_fusion(ranked)[: top_k * RERANK_POOL_MULTIPLIER]
        return self._reranker.rerank(query, pool, top_k)

    def _reference_lists(
        self, query_vector: list[float], references: QueryReferences, base: SearchFilter | None
    ) -> list[RankedList]:
        """One ranking per reference. Without a regulation, rank each regulation separately,
        so "Article 22" surfaces both the GDPR and the AI Act article."""
        return [
            RankedList(
                self._vector_store.query(query_vector, REFERENCE_CANDIDATES, scoped),
                weight=REFERENCE_LIST_WEIGHT,
            )
            for reference in references.filters()
            for scoped in _per_regulation(_combine(base, reference))
        ]


def _per_regulation(search_filter: SearchFilter) -> list[SearchFilter]:
    if search_filter.regulation is not None:
        return [search_filter]
    return [
        search_filter.model_copy(update={"regulation": regulation}) for regulation in Regulation
    ]


def _apply_query_regulation(
    search_filter: SearchFilter | None, references: QueryReferences
) -> SearchFilter | None:
    """Restrict to the one regulation the question names, unless the caller already chose."""
    named = references.single_regulation
    if named is None or (search_filter is not None and search_filter.regulation is not None):
        return search_filter
    return (search_filter or SearchFilter()).model_copy(update={"regulation": named})


def _combine(base: SearchFilter | None, reference: SearchFilter) -> SearchFilter:
    if base is None:
        return reference
    updates = {field: value for field, value in reference.model_dump().items() if value is not None}
    return base.model_copy(update=updates)
