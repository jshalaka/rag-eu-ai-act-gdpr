import pytest

from rag_eu_law.evaluation.reference_judge import (
    FactCoverage,
    ReferenceAssessment,
)
from rag_eu_law.evaluation.reference_metrics import (
    summarize_reference_coverage,
)


def _assessment(verdicts: list[str]) -> ReferenceAssessment:
    """Build test judgments; these fixtures do not test model accuracy."""
    return ReferenceAssessment(
        facts=[
            FactCoverage(
                fact_id=f"f{position}",
                verdict=verdict,
                answer_quotes=(
                    ["Example answer quotation."]
                    if verdict in {
                        "covered", "partially_covered", "contradicted"
                    }
                    else []
                ),
                reason="Example assessment reason.",
            )
            for position, verdict in enumerate(verdicts, start=1)
        ],
        reason="Fixture assessment.",
    )


def test_pilot_summary_keeps_partial_coverage_separate() -> None:
    summary = summarize_reference_coverage(
        _assessment(["covered", "partially_covered", "covered"])
    )

    assert summary["total_facts"] == 3
    assert summary["covered"] == 2
    assert summary["partially_covered"] == 1
    assert summary["full_fact_coverage"] == pytest.approx(2 / 3)


def test_all_verdicts_remain_in_the_denominator() -> None:
    summary = summarize_reference_coverage(
        _assessment([
            "covered",
            "partially_covered",
            "missing",
            "contradicted",
            "uncertain",
        ])
    )

    assert summary["total_facts"] == 5
    assert summary["full_fact_coverage"] == pytest.approx(0.2)

    for verdict in (
        "covered", "partially_covered", "missing", "contradicted", "uncertain"
    ):
        assert summary[verdict] == 1


def test_empty_assessment_has_no_coverage_score() -> None:
    summary = summarize_reference_coverage(_assessment([]))

    assert summary["total_facts"] == 0
    assert summary["full_fact_coverage"] is None


def test_all_covered_returns_full_coverage() -> None:
    summary = summarize_reference_coverage(
        _assessment(["covered", "covered"])
    )

    assert summary["full_fact_coverage"] == 1.0


def test_duplicate_fact_ids_are_rejected() -> None:
    assessment = _assessment(["covered"])
    duplicated = assessment.model_copy(
        update={"facts": [assessment.facts[0], assessment.facts[0]]}
    )

    with pytest.raises(ValueError, match="Duplicate fact IDs"):
        summarize_reference_coverage(duplicated)