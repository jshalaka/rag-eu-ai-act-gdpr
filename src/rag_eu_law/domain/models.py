"""Immutable domain models shared by every layer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_eu_law.domain.constants import (
    CHUNK_ID_SEPARATOR,
    DISCLAIMER,
    RATING_DOWN,
    RATING_UP,
)


class Regulation(StrEnum):
    GDPR = "GDPR"
    AI_ACT = "AI Act"


class UnitKind(StrEnum):
    ARTICLE = "article"
    RECITAL = "recital"
    ANNEX = "annex"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Page(_Frozen):
    """Text of one PDF page. `number` is the 1-based PDF page index."""

    source_file: str
    number: int = Field(ge=1)
    text: str


class LegalUnit(_Frozen):
    """A structural unit found by the parser, before chunking."""

    regulation: Regulation
    kind: UnitKind
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_file: str
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_title: str | None = None
    recital: str | None = None
    annex: str | None = None
    # Page number of each text line (same length as the list of lines in `text`).
    line_pages: tuple[int, ...] = ()


class Locator(_Frozen):
    """Where in a regulation a piece of text lives."""

    regulation: Regulation
    chapter: str | None = None
    article: str | None = None
    paragraph: str | None = None
    recital: str | None = None
    annex: str | None = None


class Chunk(_Frozen):
    chunk_id: str
    regulation: Regulation
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_file: str
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_title: str | None = None
    paragraph: str | None = None
    recital: str | None = None
    annex: str | None = None

    @property
    def heading(self) -> str:
        """Human-readable location, used as context for embeddings and prompts."""
        if self.article is not None:
            title = f" ({self.article_title})" if self.article_title else ""
            paragraph = f", paragraph {self.paragraph}" if self.paragraph else ""
            return f"{self.regulation.value} Article {self.article}{title}{paragraph}"
        if self.recital is not None:
            return f"{self.regulation.value} Recital {self.recital}"
        return f"{self.regulation.value} Annex {self.annex}"


class ScoredChunk(_Frozen):
    chunk: Chunk
    score: float


class SearchFilter(_Frozen):
    """Optional metadata constraints; unset fields do not filter."""

    regulation: Regulation | None = None
    article: str | None = None
    recital: str | None = None
    annex: str | None = None

    def constraints(self) -> dict[str, str]:
        """The set fields as `{metadata field: value}`."""
        values = {
            "regulation": self.regulation.value if self.regulation else None,
            "article": self.article,
            "recital": self.recital,
            "annex": self.annex,
        }
        return {field: value for field, value in values.items() if value is not None}

    def matches(self, chunk: Chunk) -> bool:
        return all(
            _chunk_field(chunk, field) == value for field, value in self.constraints().items()
        )


def _chunk_field(chunk: Chunk, field: str) -> str | None:
    value = getattr(chunk, field)
    return value.value if isinstance(value, Regulation) else value


class Citation(_Frozen):
    chunk_id: str
    regulation: Regulation
    page: int = Field(ge=1)
    article: str | None = None
    paragraph: str | None = None
    point: str | None = None
    recital: str | None = None
    annex: str | None = None

    def format(self) -> str:
        """Render as `[GDPR Art. 6(1)(a), p. 36]`."""
        return f"[{self.regulation.value} {self._locator_label()}, p. {self.page}]"

    def _locator_label(self) -> str:
        if self.article is not None:
            paragraph = f"({self.paragraph})" if self.paragraph else ""
            point = f"({self.point})" if self.point else ""
            return f"Art. {self.article}{paragraph}{point}"
        if self.recital is not None:
            return f"Recital {self.recital}"
        if self.annex is not None:
            return f"Annex {self.annex}"
        return "unknown location"


class Answer(_Frozen):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[ScoredChunk]
    latency_ms: float
    disclaimer: str = DISCLAIMER


class RunRecord(_Frozen):
    """One question asked through a user interface, kept so results can be reviewed and reused."""

    run_id: str
    created_at: str  # ISO 8601, UTC
    user: str
    question: str
    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    refused: bool
    latency_ms: float
    model: str
    index_version: str
    regulation_filter: str | None = None
    article_filter: str | None = None
    rating: int | None = None
    note: str | None = None

    @field_validator("rating")
    @classmethod
    def _rating_is_up_or_down(cls, value: int | None) -> int | None:
        if value not in (None, RATING_UP, RATING_DOWN):
            raise ValueError(f"rating must be {RATING_UP} or {RATING_DOWN}")
        return value


def build_chunk_id(locator: Locator, sequence: int = 0) -> str:
    """Deterministic id from the legal location, so re-ingestion overwrites in place.

    `sequence` disambiguates multiple chunks that share one locator.
    """
    parts = [
        locator.regulation.name,
        _tagged("art", locator.article),
        _tagged("para", locator.paragraph),
        _tagged("rec", locator.recital),
        _tagged("annex", locator.annex),
        f"seq{sequence}",
    ]
    return CHUNK_ID_SEPARATOR.join(part for part in parts if part)


def _tagged(tag: str, value: str | None) -> str:
    return f"{tag}{value}" if value else ""
