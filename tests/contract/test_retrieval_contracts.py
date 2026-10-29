"""Contract tests for Retriever and Reranker implementations."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from rag_eu_law.domain.models import Regulation, ScoredChunk, SearchFilter
from rag_eu_law.domain.protocols import Reranker, Retriever
from rag_eu_law.indexing.embedder import HashingEmbedder
from rag_eu_law.indexing.keyword_index import Bm25KeywordIndex
from rag_eu_law.indexing.pipeline import IndexingPipeline
from rag_eu_law.indexing.vector_store import InMemoryVectorStore
from rag_eu_law.retrieval.hybrid import HybridRetriever
from rag_eu_law.retrieval.reranker import CrossEncoderReranker, NoOpReranker
from tests.fixtures import SAMPLE_CHUNKS

RUN_SLOW = os.environ.get("RAG_RUN_SLOW") == "1"


def _hybrid() -> Retriever:
    embedder, store, keywords = HashingEmbedder(), InMemoryVectorStore(), Bm25KeywordIndex()
    IndexingPipeline(embedder, store, keywords).index(SAMPLE_CHUNKS)
    return HybridRetriever(embedder, store, keywords)


RETRIEVERS: list[Callable[[], Retriever]] = [_hybrid]
RERANKERS = [
    pytest.param(NoOpReranker, id="noop"),
    pytest.param(
        lambda: CrossEncoderReranker("BAAI/bge-reranker-base"),
        id="cross-encoder",
        marks=pytest.mark.skipif(not RUN_SLOW, reason="set RAG_RUN_SLOW=1 (downloads a model)"),
    ),
]


@pytest.mark.parametrize("make_retriever", RETRIEVERS)
def test_retriever_returns_at_most_top_k_without_duplicates(
    make_retriever: Callable[[], Retriever],
) -> None:
    results = make_retriever().retrieve("data subject", 3)
    ids = [r.chunk.chunk_id for r in results]
    assert len(ids) <= 3
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("make_retriever", RETRIEVERS)
def test_retriever_honours_filters(make_retriever: Callable[[], Retriever]) -> None:
    only_ai = SearchFilter(regulation=Regulation.AI_ACT)
    results = make_retriever().retrieve("data", 10, only_ai)
    assert results
    assert all(r.chunk.regulation is Regulation.AI_ACT for r in results)


@pytest.mark.parametrize("make_retriever", RETRIEVERS)
def test_retriever_is_deterministic(make_retriever: Callable[[], Retriever]) -> None:
    retriever = make_retriever()
    assert retriever.retrieve("automated decision", 3) == retriever.retrieve(
        "automated decision", 3
    )


@pytest.mark.parametrize("make_reranker", RERANKERS)
def test_reranker_returns_a_subset_of_candidates(make_reranker: Callable[[], Reranker]) -> None:
    candidates = [ScoredChunk(chunk=c, score=0.5) for c in SAMPLE_CHUNKS]
    results = make_reranker().rerank("right to erasure of personal data", candidates, 2)
    assert len(results) == 2
    assert {r.chunk.chunk_id for r in results} <= {c.chunk.chunk_id for c in candidates}


@pytest.mark.parametrize("make_reranker", RERANKERS)
def test_reranker_handles_no_candidates(make_reranker: Callable[[], Reranker]) -> None:
    assert make_reranker().rerank("anything", [], 3) == []
