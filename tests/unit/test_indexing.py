from __future__ import annotations

from pathlib import Path

import pytest

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.exceptions import ConfigurationError, IndexingError
from rag_eu_law.domain.models import Regulation, SearchFilter
from rag_eu_law.indexing.embedder import FastEmbedEmbedder, HashingEmbedder
from rag_eu_law.indexing.keyword_index import Bm25KeywordIndex
from rag_eu_law.indexing.pipeline import IndexingPipeline
from rag_eu_law.indexing.text import embedding_text, tokenize
from rag_eu_law.indexing.vector_store import InMemoryVectorStore
from tests.fixtures import SAMPLE_CHUNKS


def test_search_filter_matches_and_constraints() -> None:
    chunk = SAMPLE_CHUNKS[0]
    assert SearchFilter().matches(chunk)
    assert SearchFilter(regulation=Regulation.GDPR, article="17").matches(chunk)
    assert not SearchFilter(article="18").matches(chunk)
    assert SearchFilter(regulation=Regulation.GDPR).constraints() == {"regulation": "GDPR"}


def test_embedding_text_includes_heading() -> None:
    assert embedding_text(SAMPLE_CHUNKS[0]).startswith("GDPR Article 17 (Right to erasure)\n")


def test_tokenize_keeps_numbers() -> None:
    assert tokenize("Article 6(1)(a), GDPR!") == ["article", "6", "1", "a", "gdpr"]


def test_indexing_pipeline_fills_both_indexes_in_batches(tmp_path: Path) -> None:
    store, keywords = InMemoryVectorStore(), Bm25KeywordIndex(tmp_path / "kw.jsonl")
    pipeline = IndexingPipeline(HashingEmbedder(), store, keywords, batch_size=3)
    assert pipeline.index(SAMPLE_CHUNKS) == len(SAMPLE_CHUNKS)
    assert (store.count(), keywords.count()) == (4, 4)
    pipeline.index(SAMPLE_CHUNKS)
    assert (store.count(), keywords.count()) == (4, 4)


def test_keyword_index_reloads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "kw.jsonl"
    Bm25KeywordIndex(path).upsert(SAMPLE_CHUNKS)
    reloaded = Bm25KeywordIndex(path)
    assert reloaded.count() == len(SAMPLE_CHUNKS)
    assert reloaded.search("erasure", top_k=1)[0].chunk.article == "17"


def test_corrupt_keyword_index_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "kw.jsonl"
    path.write_text("not json\n")
    with pytest.raises(IndexingError):
        Bm25KeywordIndex(path)


def test_container_wires_configured_implementations(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, processed_dir=tmp_path, embedder="hashing", vector_store="memory"
    )
    container = Container(settings)
    assert isinstance(container.embedder, HashingEmbedder)
    assert isinstance(container.vector_store, InMemoryVectorStore)
    assert container.indexing_pipeline is container.indexing_pipeline
    assert container.ingestion_pipeline is not None


def test_container_builds_the_configured_llm_client(tmp_path: Path) -> None:
    from rag_eu_law.app.config import LLMProvider  # noqa: PLC0415
    from rag_eu_law.generation.llm_clients import OpenAIClient  # noqa: PLC0415

    settings = Settings(
        _env_file=None,
        processed_dir=tmp_path,
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="test-key",
    )
    assert isinstance(Container(settings).llm, OpenAIClient)


class FakeOnnxModel:
    embedding_size = 3

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self.seen.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_fastembed_adds_the_query_prefix_only_for_queries() -> None:
    embedder = FastEmbedEmbedder("any-model", query_prefix="Q: ")
    model = FakeOnnxModel()
    embedder._model = model
    assert embedder.dimension == 3
    embedder.embed(["document"])
    embedder.embed_query("question")
    assert model.seen == ["document", "Q: question"]


def test_fastembed_errors_are_wrapped() -> None:
    class Broken:
        def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
            raise RuntimeError("onnx failed")

    embedder = FastEmbedEmbedder("any-model")
    embedder._model = Broken()
    with pytest.raises(IndexingError, match="onnx failed"):
        embedder.embed(["x"])


def test_container_builds_fastembed_with_cache_dir(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, processed_dir=tmp_path, embedder="fastembed", embedding_cache_dir=tmp_path
    )
    assert isinstance(Container(settings).embedder, FastEmbedEmbedder)


def test_container_rejects_unknown_names(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, processed_dir=tmp_path, embedder="nope")
    with pytest.raises(ConfigurationError, match="nope"):
        _ = Container(settings).embedder
