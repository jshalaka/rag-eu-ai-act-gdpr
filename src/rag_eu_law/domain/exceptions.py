"""Custom exceptions, one family per pipeline stage."""

from __future__ import annotations


class RagError(Exception):
    """Base class for every error raised by this project."""


class ConfigurationError(RagError):
    """Settings are missing or inconsistent."""


class IngestionError(RagError):
    """Ingestion failed (loading, cleaning, parsing, chunking)."""


class SourceDocumentError(IngestionError):
    """A source PDF is missing, unreadable or contains no extractable text."""


class StructureParseError(IngestionError):
    """The legal structure of a document could not be recognised."""


class IndexingError(RagError):
    """Embedding or storing chunks failed."""


class RetrievalError(RagError):
    """Retrieval failed (empty index, bad filter, backend error)."""


class GenerationError(RagError):
    """The LLM call or answer assembly failed."""


class EvaluationError(RagError):
    """The evaluation set is missing or invalid."""


class RunStoreError(RagError):
    """Saving or reading run history failed."""


class AuthenticationError(RagError):
    """Sign-in failed."""
