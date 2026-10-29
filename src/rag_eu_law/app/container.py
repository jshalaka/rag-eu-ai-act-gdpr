"""Composition root: the only place that knows concrete classes.

Add a new embedder or vector store by writing the class and registering a factory below.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TypeVar

from rag_eu_law.app.auth import Authenticator, SupabaseAuthenticator
from rag_eu_law.app.config import LLMProvider, Settings
from rag_eu_law.app.history import HistoryService, index_version
from rag_eu_law.app.service import AskService
from rag_eu_law.domain import constants
from rag_eu_law.domain.exceptions import ConfigurationError
from rag_eu_law.domain.protocols import (
    AnswerGenerator,
    Embedder,
    KeywordIndex,
    LLMClient,
    Reranker,
    Retriever,
    RunStore,
    VectorStore,
)
from rag_eu_law.generation.generator import GroundedAnswerGenerator
from rag_eu_law.generation.llm_clients import AnthropicClient, OllamaClient, OpenAIClient
from rag_eu_law.indexing.embedder import (
    FastEmbedEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from rag_eu_law.indexing.keyword_index import Bm25KeywordIndex
from rag_eu_law.indexing.pipeline import IndexingPipeline
from rag_eu_law.indexing.vector_store import ChromaVectorStore, InMemoryVectorStore
from rag_eu_law.ingestion.chunker import LegalStructureChunker
from rag_eu_law.ingestion.cleaner import LegalTextCleaner
from rag_eu_law.ingestion.identify import SignatureRegulationDetector
from rag_eu_law.ingestion.loader import PyMuPdfLoader
from rag_eu_law.ingestion.parser import LegalStructureParser
from rag_eu_law.ingestion.pipeline import IngestionPipeline
from rag_eu_law.retrieval.hybrid import HybridRetriever
from rag_eu_law.retrieval.reranker import CrossEncoderReranker, NoOpReranker
from rag_eu_law.storage.sqlite_store import SqliteRunStore

EMBEDDER_FACTORIES: dict[str, Callable[[Settings], Embedder]] = {
    constants.EMBEDDER_SENTENCE_TRANSFORMERS: lambda s: SentenceTransformerEmbedder(
        s.embedding_model, batch_size=s.embedding_batch_size
    ),
    constants.EMBEDDER_FASTEMBED: lambda s: FastEmbedEmbedder(
        s.embedding_model,
        cache_dir=str(s.embedding_cache_dir) if s.embedding_cache_dir else None,
        batch_size=s.embedding_batch_size,
    ),
    constants.EMBEDDER_HASHING: lambda _: HashingEmbedder(),
}

VECTOR_STORE_FACTORIES: dict[str, Callable[[Settings], VectorStore]] = {
    constants.VECTOR_STORE_CHROMA: lambda s: ChromaVectorStore(
        s.processed_dir / constants.CHROMA_DIRNAME
    ),
    constants.VECTOR_STORE_MEMORY: lambda _: InMemoryVectorStore(),
}

LLM_FACTORIES: dict[LLMProvider, Callable[[Settings], LLMClient]] = {
    LLMProvider.OLLAMA: lambda s: OllamaClient(s.ollama_base_url, s.ollama_model),
    LLMProvider.ANTHROPIC: lambda s: AnthropicClient(
        s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else "", s.anthropic_model
    ),
    LLMProvider.OPENAI: lambda s: OpenAIClient(
        s.openai_api_key.get_secret_value() if s.openai_api_key else "",
        s.openai_model,
        s.openai_reasoning_effort,
    ),
}

T = TypeVar("T")


def _pick(registry: dict[str, Callable[[Settings], T]], name: str, settings: Settings) -> T:
    try:
        factory = registry[name]
    except KeyError:
        raise ConfigurationError(f"Unknown '{name}'. Available: {sorted(registry)}") from None
    return factory(settings)


class Container:
    """Builds and caches the object graph for one `Settings`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def embedder(self) -> Embedder:
        return _pick(EMBEDDER_FACTORIES, self.settings.embedder, self.settings)

    @cached_property
    def vector_store(self) -> VectorStore:
        return _pick(VECTOR_STORE_FACTORIES, self.settings.vector_store, self.settings)

    @cached_property
    def keyword_index(self) -> KeywordIndex:
        return Bm25KeywordIndex(self.settings.processed_dir / constants.KEYWORD_INDEX_FILENAME)

    @cached_property
    def ingestion_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            PyMuPdfLoader(),
            LegalTextCleaner(),
            SignatureRegulationDetector(),
            LegalStructureParser(),
            LegalStructureChunker(),
        )

    @cached_property
    def indexing_pipeline(self) -> IndexingPipeline:
        return IndexingPipeline(
            self.embedder,
            self.vector_store,
            self.keyword_index,
            batch_size=self.settings.embedding_batch_size,
        )

    @cached_property
    def reranker(self) -> Reranker:
        if self.settings.rerank_enabled:
            return CrossEncoderReranker(self.settings.rerank_model)
        return NoOpReranker()

    @cached_property
    def retriever(self) -> Retriever:
        return HybridRetriever(
            self.embedder,
            self.vector_store,
            self.keyword_index,
            reranker=self.reranker,
            candidates=self.settings.candidates_per_retriever,
        )

    @cached_property
    def llm(self) -> LLMClient:
        return LLM_FACTORIES[self.settings.llm_provider](self.settings)

    @cached_property
    def answer_generator(self) -> AnswerGenerator:
        return GroundedAnswerGenerator(self.llm)

    @cached_property
    def ask_service(self) -> AskService:
        return AskService(self.retriever, self.answer_generator, self.settings.top_k)

    @cached_property
    def run_store(self) -> RunStore:
        """The local run store. In Supabase mode runs go through a signed-in user's store."""
        return SqliteRunStore(self.settings.history_db_path)

    @cached_property
    def authenticator(self) -> Authenticator | None:
        settings = self.settings
        if settings.run_store != constants.RUN_STORE_SUPABASE:
            return None
        key = settings.supabase_anon_key
        return SupabaseAuthenticator(
            settings.supabase_url or "", key.get_secret_value() if key else ""
        )

    def history_service_for(self, run_store: RunStore) -> HistoryService:
        chunks_path = self.settings.processed_dir / constants.CHUNKS_FILENAME
        return HistoryService(
            self.ask_service, run_store, self.settings.llm_label, index_version(chunks_path)
        )

    @cached_property
    def history_service(self) -> HistoryService:
        if self.authenticator is not None:
            raise ConfigurationError(
                "Supabase mode needs a signed-in user: use history_service_for"
            )
        return self.history_service_for(self.run_store)
