"""Aggregate metrics over question results."""

from __future__ import annotations

from collections.abc import Callable

from rag_eu_law.evaluation.models import (
    EvalSummary,
    QuestionResult,
    QuestionType,
    TypeSummary,
)


def summarize(results: list[QuestionResult]) -> EvalSummary:
    answerable = [r for r in results if r.type is not QuestionType.OUT_OF_SCOPE]
    refusable = [r for r in results if r.type is QuestionType.OUT_OF_SCOPE]
    return EvalSummary(
        answerable_questions=len(answerable),
        out_of_scope_questions=len(refusable),
        mean_retrieval_latency_ms=_mean(
            [r.retrieval_latency_ms for r in results if r.error is None]
        ),
        retrieval_hit_rate_at_k=_rate(answerable, lambda r: r.retrieval_hit is True),
        complete_source_retrieval_rate_at_k=_rate(
            answerable,
            lambda r: (
                r.error is None
                and r.retrieval_recall == 1.0
                and r.missing_sources == []
            ),
        ),
        mean_retrieval_recall=_mean([r.retrieval_recall for r in answerable]),
        citation_accuracy=_rate(
            answerable, lambda r: r.citation_correct is True, needs_answer=True
        ),
        false_refusal_rate=_rate(answerable, lambda r: r.refused is True, needs_answer=True),
        refusal_accuracy=_rate(refusable, lambda r: r.refused is True, needs_answer=True),
        mean_latency_ms=_mean([r.latency_ms for r in results]),
        by_type={
            kind.value: _type_summary([r for r in results if r.type is kind])
            for kind in QuestionType
            if kind is not QuestionType.OUT_OF_SCOPE
        },
    )


def _rate(
    results: list[QuestionResult],
    predicate: Callable[[QuestionResult], bool],
    needs_answer: bool = False,
) -> float | None:
    """Share of `results` satisfying `predicate`; None when there is nothing to measure."""
    pool = [r for r in results if not needs_answer or r.refused is not None or r.error]
    if not pool or (needs_answer and all(r.refused is None for r in pool)):
        return None
    return sum(predicate(r) for r in pool) / len(pool)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _type_summary(results: list[QuestionResult]) -> TypeSummary:
    return TypeSummary(
        questions=len(results),
        retrieval_hit_rate=_rate(results, lambda r: r.retrieval_hit is True),
    )
