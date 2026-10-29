"""Compare aggregate and per-question retrieval against a selected baseline."""

import math

from rag_eu_law.evaluation.metrics import summarize
from rag_eu_law.evaluation.models import QuestionResult, QuestionType


TOLERANCE = 1e-12

METRIC_NAMES = (
    "retrieval_hit_rate_at_k",
    "complete_source_retrieval_rate_at_k",
    "mean_retrieval_recall",
)


def _questions(results: list[QuestionResult]) -> dict:
    """Identify the question population independently of record order."""
    questions = {}

    for result in results:
        if result.question_id in questions:
            raise ValueError(f"Duplicate question ID: {result.question_id}")

        questions[result.question_id] = (
            result.type.value,
            result.question,
        )

    return questions


def _has_retrieval_measurements(results: list[QuestionResult]) -> bool:
    """Reject missing or invalid measurements before comparing scores."""
    answerable = [
        result
        for result in results
        if result.type is not QuestionType.OUT_OF_SCOPE
    ]

    return bool(answerable) and all(
        result.retrieval_hit is not None
        and result.retrieval_recall is not None
        and math.isfinite(result.retrieval_recall)
        and 0 <= result.retrieval_recall <= 1
        and result.missing_sources is not None
        for result in answerable
    )


def _complete(result: QuestionResult) -> bool:
    """Use the same complete-source definition as the summary metric."""
    return (
        result.error is None
        and result.retrieval_recall == 1.0
        and result.missing_sources == []
    )


def _question_regressions(
    baseline: list[QuestionResult],
    candidate: list[QuestionResult],
) -> list[dict]:
    """Find individual declines that aggregate improvements could hide."""
    candidate_by_id = {
        result.question_id: result
        for result in candidate
    }
    regressions = []

    for previous in baseline:
        if previous.type is QuestionType.OUT_OF_SCOPE:
            continue

        current = candidate_by_id[previous.question_id]
        previous_recall = previous.retrieval_recall
        current_recall = current.retrieval_recall

        # Measurements are checked by the caller before this comparison.
        if previous_recall is None or current_recall is None:
            raise ValueError("Per-question comparison requires recall values")

        declined = []

        if previous.retrieval_hit is True and current.retrieval_hit is False:
            declined.append("retrieval_hit")

        if _complete(previous) and not _complete(current):
            declined.append("complete_source_retrieval")

        if current_recall < previous_recall - TOLERANCE:
            declined.append("retrieval_recall")

        if declined:
            regressions.append({
                "question_id": previous.question_id,
                "declined_measures": declined,
                "baseline_recall": previous_recall,
                "candidate_recall": current_recall,
                "recall_delta": current_recall - previous_recall,
            })

    return regressions


def compare_retrieval_runs(
    baseline: list[QuestionResult],
    candidate: list[QuestionResult],
) -> dict:
    """Apply aggregate and per-question non-decline checks."""
    baseline_questions = _questions(baseline)
    candidate_questions = _questions(candidate)

    # A baseline must contain usable measurements and successful executions.
    if not _has_retrieval_measurements(baseline):
        raise ValueError("Baseline lacks valid retrieval measurements")

    if any(result.error is not None for result in baseline):
        raise ValueError("Baseline contains application errors")

    checks = {
        "same_questions": baseline_questions == candidate_questions,
        "no_application_errors": all(
            result.error is None for result in candidate
        ),
        "retrieval_measurements_present": _has_retrieval_measurements(candidate),
    }

    # Avoid comparing incompatible or incomplete question populations.
    if not all(checks.values()):
        return {
            "passed": False,
            "checks": checks,
            "metrics": {},
            "question_regressions": [],
        }

    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    metrics = {}

    for name in METRIC_NAMES:
        previous = getattr(baseline_summary, name)
        current = getattr(candidate_summary, name)

        # Allow floating-point noise without allowing a meaningful decline.
        no_decline = (
            previous is not None
            and current is not None
            and math.isfinite(previous)
            and math.isfinite(current)
            and current >= previous - TOLERANCE
        )

        checks[f"{name}_not_lower"] = no_decline
        metrics[name] = {
            "baseline": previous,
            "candidate": current,
            "delta": (
                current - previous
                if current is not None and previous is not None
                else None
            ),
        }

    question_regressions = _question_regressions(baseline, candidate)
    checks["per_question_retrieval_not_lower"] = not question_regressions

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "question_regressions": question_regressions,
    }