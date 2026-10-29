"""Evaluation data: questions, per-question results and the metric summary."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from rag_eu_law.domain.models import Chunk, Citation, Regulation


class QuestionType(StrEnum):
    DIRECT_LOOKUP = "direct_lookup"
    CONCEPT = "concept"
    CROSS_REGULATION = "cross_regulation"
    OUT_OF_SCOPE = "out_of_scope"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ExpectedSource(_Frozen):
    """A source that supports the answer: one article, recital or annex of a regulation."""

    regulation: Regulation
    article: str | None = None
    recital: str | None = None
    annex: str | None = None

    @field_validator("article", "recital", "annex", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        return None if value is None else str(value)

    @model_validator(mode="after")
    def _exactly_one_locator(self) -> ExpectedSource:
        if sum(v is not None for v in (self.article, self.recital, self.annex)) != 1:
            raise ValueError("give exactly one of article, recital, annex")
        return self

    def matches(self, item: Chunk | Citation) -> bool:
        return (
            item.regulation is self.regulation
            and item.article == self.article
            and item.recital == self.recital
            and item.annex == self.annex
        )


class EvalQuestion(_Frozen):
    id: str
    type: QuestionType
    question: str
    expected: list[ExpectedSource] = []

    @model_validator(mode="after")
    def _expected_matches_type(self) -> EvalQuestion:
        refused = self.type is QuestionType.OUT_OF_SCOPE
        if refused and self.expected:
            raise ValueError(f"{self.id}: out_of_scope questions must not list expected sources")
        if not refused and not self.expected:
            raise ValueError(f"{self.id}: answerable questions need at least one expected source")
        return self

    @property
    def answerable(self) -> bool:
        return self.type is not QuestionType.OUT_OF_SCOPE


class QuestionResult(_Frozen):
    question_id: str
    type: QuestionType
    question: str
    retrieved: list[str]
    retrieval_recall: float | None = None
    retrieval_hit: bool | None = None
    missing_sources: list[ExpectedSource] | None = None
    answer: str | None = None
    citations: list[str] = []
    refused: bool | None = None
    citation_correct: bool | None = None
    retrieval_latency_ms: float | None = None
    retrieved_context: list[Chunk] | None = None
    latency_ms: float | None = None
    error: str | None = None


class TypeSummary(_Frozen):
    questions: int
    retrieval_hit_rate: float | None


class EvalSummary(_Frozen):
    answerable_questions: int
    out_of_scope_questions: int
    retrieval_hit_rate_at_k: float | None
    complete_source_retrieval_rate_at_k: float | None = None
    mean_retrieval_recall: float | None
    citation_accuracy: float | None
    false_refusal_rate: float | None
    refusal_accuracy: float | None
    mean_latency_ms: float | None
    mean_retrieval_latency_ms: float | None = None
    by_type: dict[str, TypeSummary]
