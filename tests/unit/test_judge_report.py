import pytest

from rag_eu_law.evaluation.judge_report import render_judge_report
from rag_eu_law.evaluation.models import QuestionResult, QuestionType


def _result(
    qid: str,
    *,
    out_of_scope: bool = False,
    refused: bool = False,
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        type=(
            QuestionType.OUT_OF_SCOPE
            if out_of_scope else QuestionType.CONCEPT
        ),
        question="Example question?",
        retrieved=[],
        retrieval_recall=None if out_of_scope else 0.5,
        refused=refused,
    )


def _record(qid: str, verdicts: list[str]) -> dict:
    return {
        "question_id": qid,
        "status": "assessed",
        "assessment": {
            "answer_behavior": "answered",
            "relevance": "relevant",
            "claims": [
                {
                    "claim": f"Claim {index}",
                    "verdict": verdict,
                    "evidence_chunk_ids": (
                        ["chunk-1"]
                        if verdict in {"supported", "contradicted"}
                        else []
                    ),
                    "reason": "Example assessment reason.",
                }
                for index, verdict in enumerate(verdicts)
            ],
            "reason": "Example overall assessment.",
        },
    }


def test_report_calculates_support_rate_and_lists_review_claims() -> None:
    report = render_judge_report(
        [_result("q1")],
        [_record("q1", ["supported", "insufficient_evidence"])],
    )

    assert "claim support rate: 50.0%" in report
    assert "| supported | 1 |" in report
    assert "| insufficient_evidence | 1 |" in report
    assert "| q1 | 50% | answered | assessed | 1 / 2 |" in report
    assert "Claim: Claim 1" in report
    assert "Human review: pending." in report


def test_report_distinguishes_correct_and_false_refusals() -> None:
    report = render_judge_report(
        [
            _result("answerable", refused=True),
            _result("outside", out_of_scope=True, refused=True),
        ],
        [
            _record("answerable", []),
            _record("outside", []),
        ],
    )

    assert "| answerable | 50% | false refusal |" in report
    assert "| outside | n/a | correct refusal |" in report
    assert "claim support rate: n/a" in report


def test_report_keeps_judge_errors_out_of_claim_counts() -> None:
    report = render_judge_report(
        [_result("q1"), _result("q2")],
        [
            _record("q1", ["supported"]),
            {
                "question_id": "q2",
                "status": "judge_error",
                "assessment": None,
            },
        ],
    )

    assert "- Assessed: 1" in report
    assert "- Judge errors: 1" in report
    assert "- Claims assessed: 1" in report
    assert "| q2 | 50% | answered | judge_error | n/a |" in report


def test_report_rejects_mismatched_question_ids() -> None:
    with pytest.raises(ValueError, match="must match"):
        render_judge_report(
            [_result("q1")],
            [_record("different", [])],
        )


@pytest.mark.parametrize("duplicate_side", ["application", "judge"])
def test_report_rejects_duplicate_question_ids(
    duplicate_side: str,
) -> None:
    results = [_result("q1")]
    records = [_record("q1", [])]

    if duplicate_side == "application":
        results.append(_result("q1"))
    else:
        records.append(_record("q1", []))

    with pytest.raises(ValueError, match="Duplicate"):
        render_judge_report(results, records)


def test_empty_report_has_no_numeric_support_score() -> None:
    report = render_judge_report([], [])

    assert "- Application results: 0" in report
    assert "- Claims assessed: 0" in report
    assert "claim support rate: n/a" in report


def _review(**updates) -> dict:
    """Create a completed review linked to the first claim of q1."""
    human_review = {
        "status": "reviewed",
        "decision": "agree_with_judge",
        "finding_category": "grounding",
        "evidence_chunk_ids": [],
        "reason": "The supplied context does not establish this claim.",
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-10-25",
    }
    human_review.update(updates)

    return {
        "review_id": "q1:claim-1",
        "human_review": human_review,
    }


def test_report_displays_completed_human_review() -> None:
    report = render_judge_report(
        [_result("q1")],
        [_record("q1", ["insufficient_evidence"])],
        reviews=[_review()],
    )

    assert "Human review: reviewed." in report
    assert "- Decision: agree_with_judge" in report
    assert "- Reviewer: Test reviewer" in report
    assert "Human review: pending." not in report

    # Human agreement must not rewrite the original judge verdict.
    assert "| insufficient_evidence | 1 |" in report


def test_report_keeps_reviewed_uncertainty_visible() -> None:
    report = render_judge_report(
        [_result("q1")],
        [_record("q1", ["insufficient_evidence"])],
        reviews=[_review(decision="needs_more_evidence")],
    )

    assert "Human review: reviewed." in report
    assert "Unresolved — further adjudication required" in report


def test_report_displays_pending_review() -> None:
    report = render_judge_report(
        [_result("q1")],
        [_record("q1", ["insufficient_evidence"])],
        reviews=[_review(status="pending", decision=None)],
    )

    assert "Human review: pending." in report


def test_review_uses_original_claim_position() -> None:
    review = _review()
    review["review_id"] = "q1:claim-2"

    report = render_judge_report(
        [_result("q1")],
        [_record("q1", ["supported", "insufficient_evidence"])],
        reviews=[review],
    )

    assert "Review ID: q1:claim-2" in report
    assert "Human review: reviewed." in report


def test_duplicate_review_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate review ID"):
        render_judge_report(
            [_result("q1")],
            [_record("q1", ["insufficient_evidence"])],
            reviews=[_review(), _review()],
        )


def test_review_for_unknown_claim_is_rejected() -> None:
    review = _review()
    review["review_id"] = "q1:claim-99"

    with pytest.raises(ValueError, match="does not match"):
        render_judge_report(
            [_result("q1")],
            [_record("q1", ["insufficient_evidence"])],
            reviews=[review],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "unknown"}, "Unknown human-review status"),
        ({"decision": None}, "valid decision"),
        ({"reason": ""}, "requires reason"),
        ({"reviewer": None}, "requires reviewer"),
        ({"reviewed_at": ""}, "requires reviewed_at"),
        ({"finding_category": None}, "requires finding_category"),
    ],
)
def test_incomplete_review_is_rejected(
    updates: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        render_judge_report(
            [_result("q1")],
            [_record("q1", ["insufficient_evidence"])],
            reviews=[_review(**updates)],
        )