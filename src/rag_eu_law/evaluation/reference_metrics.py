"""Summarise reference-fact coverage without changing judge verdicts."""

from collections import Counter

from rag_eu_law.evaluation.reference_judge import ReferenceAssessment


def summarize_reference_coverage(
    assessment: ReferenceAssessment,
) -> dict:
    """Count verdicts and calculate strict full-fact coverage."""
    fact_ids = [fact.fact_id for fact in assessment.facts]

    # Duplicate facts would distort the denominator.
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("Duplicate fact IDs in coverage assessment")

    counts = Counter(fact.verdict for fact in assessment.facts)
    total = len(assessment.facts)

    return {
        "total_facts": total,
        "covered": counts["covered"],
        "partially_covered": counts["partially_covered"],
        "missing": counts["missing"],
        "contradicted": counts["contradicted"],
        "uncertain": counts["uncertain"],
        # No facts means no measurable coverage, rather than a perfect score.
        "full_fact_coverage": counts["covered"] / total if total else None,
    }
