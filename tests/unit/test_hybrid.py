from __future__ import annotations

import pytest

from rag_eu_law.domain.exceptions import RetrievalError
from rag_eu_law.domain.models import Chunk, Regulation, ScoredChunk, SearchFilter
from rag_eu_law.domain.protocols import Reranker
from rag_eu_law.indexing.embedder import HashingEmbedder
from rag_eu_law.indexing.keyword_index import Bm25KeywordIndex
from rag_eu_law.indexing.pipeline import IndexingPipeline
from rag_eu_law.indexing.vector_store import InMemoryVectorStore
from rag_eu_law.retrieval.hybrid import HybridRetriever
from tests.fixtures import SAMPLE_CHUNKS, make_chunk

FILLER = [
    make_chunk(f"GDPR:art{n}:seq0", "processing of personal data by the controller", article=str(n))
    for n in (30, 31, 32, 33)
]


def _build(
    chunks: list[Chunk] = SAMPLE_CHUNKS, reranker: Reranker | None = None
) -> HybridRetriever:
    embedder, store, keywords = HashingEmbedder(), InMemoryVectorStore(), Bm25KeywordIndex()
    IndexingPipeline(embedder, store, keywords).index(chunks)
    return HybridRetriever(embedder, store, keywords, reranker=reranker)


def _ids(results: list[ScoredChunk]) -> list[str]:
    return [item.chunk.chunk_id for item in results]


def test_named_article_is_ranked_first_even_when_text_does_not_match() -> None:
    retriever = _build([*SAMPLE_CHUNKS, *FILLER])
    results = retriever.retrieve("What does Article 22 say about processing of personal data?", 3)
    assert _ids(results)[0] == "GDPR:art22:seq0"


def test_named_regulation_restricts_results() -> None:
    results = _build().retrieve("Under the AI Act, which data rules apply?", 10)
    assert {r.chunk.regulation for r in results} == {Regulation.AI_ACT}


def test_both_regulations_named_searches_everything() -> None:
    results = _build().retrieve("How do the AI Act and the GDPR treat biometric data?", 10)
    assert {r.chunk.regulation for r in results} == {Regulation.AI_ACT, Regulation.GDPR}


def test_caller_filter_wins_over_query_regulation() -> None:
    gdpr_only = SearchFilter(regulation=Regulation.GDPR)
    results = _build().retrieve("Under the AI Act, what is prohibited?", 10, gdpr_only)
    assert {r.chunk.regulation for r in results} == {Regulation.GDPR}


def test_caller_filter_combines_with_article_reference() -> None:
    results = _build().retrieve(
        "Article 5 prohibited", 10, SearchFilter(regulation=Regulation.GDPR)
    )
    assert "AI_ACT:art5:seq0" not in _ids(results)


def test_recital_reference_is_found() -> None:
    assert _ids(_build().retrieve("What does Recital 54 say?", 2))[0] == "AI_ACT:rec54:seq0"


def test_results_are_limited_and_sorted() -> None:
    results = _build().retrieve("data subject right", 2)
    assert len(results) <= 2
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_reranker_is_applied_to_the_fused_pool() -> None:
    class ReversingReranker:
        def rerank(
            self, query: str, candidates: list[ScoredChunk], top_k: int
        ) -> list[ScoredChunk]:
            return list(reversed(candidates))[:top_k]

    plain = _build().retrieve("data subject right", 4)
    reranked = _build(reranker=ReversingReranker()).retrieve("data subject right", 4)
    assert _ids(reranked) != _ids(plain)


def test_empty_index_and_bad_top_k_are_errors() -> None:
    empty = HybridRetriever(HashingEmbedder(), InMemoryVectorStore(), Bm25KeywordIndex())
    with pytest.raises(RetrievalError, match="empty"):
        empty.retrieve("anything", 3)
    with pytest.raises(RetrievalError, match="top_k"):
        _build().retrieve("anything", 0)


def test_unscoped_article_reference_surfaces_both_regulations() -> None:
    same_number = [
        make_chunk("GDPR:art5:seq0", "lawfulness principles of processing", article="5"),
        make_chunk(
            "AI_ACT:art5:seq0", "prohibited practices", regulation=Regulation.AI_ACT, article="5"
        ),
    ]
    results = _build([*same_number, *FILLER]).retrieve("What does Article 5 say?", 2)
    assert {r.chunk.regulation for r in results} == {Regulation.GDPR, Regulation.AI_ACT}
