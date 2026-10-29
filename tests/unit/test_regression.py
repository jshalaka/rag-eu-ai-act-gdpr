"""Test aggregate and per-question retrieval regression checks."""

import pytest

from rag_eu_law.evaluation.models import QuestionResult, QuestionType
from rag_eu_law.evaluation.regression import compare_retrieval_runs


def _result(qid: str = "q1", **updates) -> QuestionResult:
    """Create a measured retrieval result for isolated gate testing."""
    values = {
        "question_id": qid,
        "type": QuestionType.CONCEPT,
        "question": "Example question?",
        "retrieved": ["example-chunk"],
        "retrieval_hit": True,
        "retrieval_recall": 1.0,
        "missing_sources": [],
    }
    values.update(updates)
    return QuestionResult.model_validate(values)


def test_unchanged_run_passes() -> None:
    baseline = [_result()]

    report = compare_retrieval_runs(baseline, baseline)

    assert report["passed"] is True
    assert all(report["checks"].values())


def test_record_order_does_not_change_comparison() -> None:
    baseline = [_result("q1"), _result("q2")]

    report = compare_retrieval_runs(baseline, list(reversed(baseline)))

    assert report["passed"] is True


def test_lower_recall_fails_even_when_hit_rate_is_unchanged() -> None:
    # A retrieval hit can remain true while expected-source coverage declines.
    report = compare_retrieval_runs(
        [_result()],
        [_result(retrieval_recall=0.5)],
    )

    assert report["passed"] is False
    assert report["checks"]["retrieval_hit_rate_at_k_not_lower"] is True
    assert report["checks"]["mean_retrieval_recall_not_lower"] is False
    assert (
        report["checks"]["complete_source_retrieval_rate_at_k_not_lower"]
        is False
    )


def test_lower_hit_rate_fails() -> None:
    candidate = [_result(retrieval_hit=False, retrieval_recall=0.0)]

    report = compare_retrieval_runs([_result()], candidate)

    assert report["passed"] is False
    assert report["checks"]["retrieval_hit_rate_at_k_not_lower"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"question_id": "different"},
        {"question": "Changed question?"},
        {"type": QuestionType.DIRECT_LOOKUP},
    ],
)
def test_changed_question_population_fails(updates: dict) -> None:
    report = compare_retrieval_runs([_result()], [_result(**updates)])

    assert report["passed"] is False
    assert report["checks"]["same_questions"] is False
    assert report["metrics"] == {}


def test_application_error_fails() -> None:
    report = compare_retrieval_runs(
        [_result()],
        [_result(error="Retrieval failed")],
    )

    assert report["passed"] is False
    assert report["checks"]["no_application_errors"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"retrieval_hit": None},
        {"retrieval_recall": None},
        {"missing_sources": None},
    ],
)
def test_missing_candidate_measurements_fail(updates: dict) -> None:
    report = compare_retrieval_runs([_result()], [_result(**updates)])

    assert report["passed"] is False
    assert report["checks"]["retrieval_measurements_present"] is False


def test_empty_candidate_fails() -> None:
    report = compare_retrieval_runs([_result()], [])

    assert report["passed"] is False


def test_invalid_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="Baseline lacks"):
        compare_retrieval_runs([], [_result()])


def test_baseline_with_errors_is_rejected() -> None:
    with pytest.raises(ValueError, match="Baseline contains"):
        compare_retrieval_runs(
            [_result(error="Index unavailable")],
            [_result()],
        )


@pytest.mark.parametrize("duplicate_side", ["baseline", "candidate"])
def test_duplicate_ids_are_rejected(duplicate_side: str) -> None:
    baseline = [_result()]
    candidate = [_result()]

    # Duplicate IDs must not silently overwrite a question.
    if duplicate_side == "baseline":
        baseline.append(_result())
    else:
        candidate.append(_result())

    with pytest.raises(ValueError, match="Duplicate question ID"):
        compare_retrieval_runs(baseline, candidate)


def test_question_decline_fails_when_aggregate_scores_are_unchanged() -> None:
    baseline = [
        _result("q1"),
        _result(
            "q2",
            retrieved=[],
            retrieval_hit=False,
            retrieval_recall=0.0,
            missing_sources=[{"regulation": "GDPR", "article": "17"}],
        ),
    ]

    # Swapping success between questions preserves every aggregate score.
    candidate = [
        baseline[0].model_copy(update={"question_id": "q2"}),
        baseline[1].model_copy(update={"question_id": "q1"}),
    ]

    report = compare_retrieval_runs(baseline, candidate)

    assert all(
        report["checks"][f"{name}_not_lower"]
        for name in (
            "retrieval_hit_rate_at_k",
            "complete_source_retrieval_rate_at_k",
            "mean_retrieval_recall",
        )
    )
    assert report["passed"] is False
    assert report["checks"]["per_question_retrieval_not_lower"] is False
    assert [
        item["question_id"]
        for item in report["question_regressions"]
    ] == ["q1"]


def test_question_improvement_is_allowed() -> None:
    baseline = [
        _result(
            "q1",
            retrieval_recall=0.5,
            missing_sources=[{"regulation": "GDPR", "article": "17"}],
        )
    ]

    report = compare_retrieval_runs(baseline, [_result("q1")])

    assert report["passed"] is True
    assert report["question_regressions"] == []