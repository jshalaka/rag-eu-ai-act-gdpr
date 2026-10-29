"""Contract tests: every Embedder, VectorStore and KeywordIndex must pass these."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from rag_eu_law.domain.exceptions import IndexingError
from rag_eu_law.domain.models import Regulation, SearchFilter
from rag_eu_law.domain.protocols import Embedder, KeywordIndex, VectorStore
from rag_eu_law.indexing.embedder import (
    FastEmbedEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from rag_eu_law.indexing.keyword_index import Bm25KeywordIndex
from rag_eu_law.indexing.text import embedding_text
from rag_eu_law.indexing.vector_store import ChromaVectorStore, InMemoryVectorStore
from tests.fixtures import SAMPLE_CHUNKS

RUN_SLOW = os.environ.get("RAG_RUN_SLOW") == "1"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EMBEDDER_FACTORIES: list[object] = [
    pytest.param(HashingEmbedder, id="hashing"),
    pytest.param(
        lambda: FastEmbedEmbedder(EMBEDDING_MODEL),
        id="fastembed",
        marks=pytest.mark.skipif(not RUN_SLOW, reason="set RAG_RUN_SLOW=1 (downloads a model)"),
    ),
    pytest.param(
        lambda: SentenceTransformerEmbedder(EMBEDDING_MODEL),
        id="sentence-transformers",
        marks=pytest.mark.skipif(not RUN_SLOW, reason="set RAG_RUN_SLOW=1 (downloads a model)"),
    ),
]


@pytest.fixture(params=EMBEDDER_FACTORIES)
def embedder(request: pytest.FixtureRequest) -> Embedder:
    factory: Callable[[], Embedder] = request.param
    return factory()


@pytest.fixture(params=["memory", "chroma"])
def vector_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[VectorStore]:
    if request.param == "memory":
        yield InMemoryVectorStore()
    else:
        yield ChromaVectorStore(tmp_path / "chroma", collection_name=f"c{id(tmp_path)}")


@pytest.fixture(params=["memory-only", "persistent"])
def keyword_index(request: pytest.FixtureRequest, tmp_path: Path) -> KeywordIndex:
    path = tmp_path / "kw.jsonl" if request.param == "persistent" else None
    return Bm25KeywordIndex(path)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# --- Embedder ----------------------------------------------------------------


def test_embedder_returns_one_vector_per_text(embedder: Embedder) -> None:
    vectors = embedder.embed(["right to erasure", "automated decisions", "biometric data"])
    assert len(vectors) == 3
    assert all(len(vector) == embedder.dimension for vector in vectors)


def test_embedder_is_deterministic(embedder: Embedder) -> None:
    first = embedder.embed(["right to erasure"])[0]
    second = embedder.embed(["right to erasure"])[0]
    assert first == pytest.approx(second)


def test_embedder_vectors_are_unit_length(embedder: Embedder) -> None:
    assert _norm(embedder.embed(["right to erasure"])[0]) == pytest.approx(1.0, abs=1e-4)
    assert _norm(embedder.embed_query("right to erasure")) == pytest.approx(1.0, abs=1e-4)


def test_embedder_handles_empty_batch(embedder: Embedder) -> None:
    assert embedder.embed([]) == []


# --- VectorStore ---------------------------------------------------------------


def _fill(store: VectorStore, embedder: HashingEmbedder) -> None:
    store.upsert(SAMPLE_CHUNKS, embedder.embed([embedding_text(c) for c in SAMPLE_CHUNKS]))


def test_vector_store_finds_most_similar_chunk(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    results = vector_store.query(embedder.embed_query("erasure of personal data"), top_k=2)
    assert results[0].chunk.chunk_id == "GDPR:art17:seq0"
    assert results[0].score >= results[1].score


def test_vector_store_upsert_is_idempotent(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    _fill(vector_store, embedder)
    assert vector_store.count() == len(SAMPLE_CHUNKS)


def test_vector_store_respects_top_k(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    assert len(vector_store.query(embedder.embed_query("data"), top_k=1)) == 1


def test_vector_store_filters_by_regulation(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    only_ai = SearchFilter(regulation=Regulation.AI_ACT)
    results = vector_store.query(embedder.embed_query("data"), top_k=10, search_filter=only_ai)
    assert {r.chunk.regulation for r in results} == {Regulation.AI_ACT}


def test_vector_store_filters_by_article(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    article = SearchFilter(regulation=Regulation.GDPR, article="22")
    results = vector_store.query(embedder.embed_query("data"), top_k=10, search_filter=article)
    assert [r.chunk.chunk_id for r in results] == ["GDPR:art22:seq0"]


def test_vector_store_empty_store_and_no_match(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    assert vector_store.query(embedder.embed_query("data"), top_k=3) == []
    _fill(vector_store, embedder)
    missing = SearchFilter(article="999")
    assert vector_store.query(embedder.embed_query("data"), 3, search_filter=missing) == []


def test_vector_store_rejects_mismatched_lengths(vector_store: VectorStore) -> None:
    with pytest.raises(IndexingError):
        vector_store.upsert(SAMPLE_CHUNKS, [[0.1, 0.2]])


def test_vector_store_roundtrips_chunk_metadata(vector_store: VectorStore) -> None:
    embedder = HashingEmbedder()
    _fill(vector_store, embedder)
    top = vector_store.query(embedder.embed_query("subliminal manipulation"), top_k=1)[0].chunk
    assert top == SAMPLE_CHUNKS[2]


# --- KeywordIndex ----------------------------------------------------------------


def test_keyword_index_finds_article_by_number(keyword_index: KeywordIndex) -> None:
    keyword_index.upsert(SAMPLE_CHUNKS)
    results = keyword_index.search("Article 22", top_k=3)
    assert results[0].chunk.chunk_id == "GDPR:art22:seq0"


def test_keyword_index_finds_recital_by_number(keyword_index: KeywordIndex) -> None:
    keyword_index.upsert(SAMPLE_CHUNKS)
    assert keyword_index.search("recital 54", top_k=1)[0].chunk.recital == "54"


def test_keyword_index_filters(keyword_index: KeywordIndex) -> None:
    keyword_index.upsert(SAMPLE_CHUNKS)
    results = keyword_index.search(
        "data", top_k=10, search_filter=SearchFilter(regulation=Regulation.AI_ACT)
    )
    assert results
    assert {r.chunk.regulation for r in results} == {Regulation.AI_ACT}


def test_keyword_index_upsert_is_idempotent_and_overwrites(keyword_index: KeywordIndex) -> None:
    keyword_index.upsert(SAMPLE_CHUNKS)
    changed = SAMPLE_CHUNKS[0].model_copy(update={"text": "Completely different wording."})
    keyword_index.upsert([changed])
    results = keyword_index.search("different wording", top_k=5)
    assert [r.chunk.chunk_id for r in results] == ["GDPR:art17:seq0"]
    assert len(keyword_index.search("data", top_k=10)) <= len(SAMPLE_CHUNKS)


def test_keyword_index_no_match_and_empty_query(keyword_index: KeywordIndex) -> None:
    assert keyword_index.search("anything", top_k=3) == []
    keyword_index.upsert(SAMPLE_CHUNKS)
    assert keyword_index.search("zzzunknownzzz", top_k=3) == []
    assert keyword_index.search("   ", top_k=3) == []


def test_keyword_index_respects_top_k(keyword_index: KeywordIndex) -> None:
    keyword_index.upsert(SAMPLE_CHUNKS)
    assert len(keyword_index.search("data", top_k=1)) == 1
