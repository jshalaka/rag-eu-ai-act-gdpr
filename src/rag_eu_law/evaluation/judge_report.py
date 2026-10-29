"""Render application results, judge assessments, and human reviews."""

from collections import Counter

from rag_eu_law.evaluation.judge_models import JudgeAssessment
from rag_eu_law.evaluation.models import QuestionResult, QuestionType


def _cell(value: object) -> str:
    """Escape table separators and flatten text for Markdown display."""
    return str(value).replace("|", r"\|").replace("\n", " ")


def _refusal_outcome(result: QuestionResult) -> str:
    """Compare the recorded refusal with the dataset's expected scope."""
    # An execution failure is not a refusal decision.
    if result.error is not None:
        return "application error"

    # Retrieval-only runs do not measure refusal behaviour.
    if result.refused is None:
        return "not measured"

    out_of_scope = result.type is QuestionType.OUT_OF_SCOPE

    if result.refused:
        return "correct refusal" if out_of_scope else "false refusal"

    return "unexpected answer" if out_of_scope else "answered"


def _human_review_lines(review: dict | None) -> list[str]:
    """Display human review without changing the original judge verdict."""
    if review is None or review.get("status") == "pending":
        return ["Human review: pending."]

    if review.get("status") != "reviewed":
        raise ValueError("Unknown human-review status")

    allowed_decisions = {
        "agree_with_judge",
        "disagree_with_judge",
        "needs_more_evidence",
    }

    if review.get("decision") not in allowed_decisions:
        raise ValueError("Reviewed claim requires a valid decision")

    # Completed reviews require an explanation and reviewer attribution.
    for field in ("reason", "reviewer", "reviewed_at", "finding_category"):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Reviewed claim requires {field}")

    # A reviewed case can still require further adjudication.
    resolution = (
        "Unresolved — further adjudication required"
        if review["decision"] == "needs_more_evidence"
        else "Decision recorded"
    )

    evidence = ", ".join(review.get("evidence_chunk_ids", [])) or "none"

    return [
        "Human review: reviewed.",
        "",
        f"- Decision: {_cell(review['decision'])}",
        f"- Resolution: {resolution}",
        f"- Category: {_cell(review['finding_category'])}",
        f"- Reason: {_cell(review['reason'])}",
        f"- Reviewer: {_cell(review['reviewer'])}",
        f"- Review date: {_cell(review['reviewed_at'])}",
        f"- Evidence IDs: {_cell(evidence)}",
    ]


def render_judge_report(
    results: list[QuestionResult],
    records: list[dict],
    reviews: list[dict] | None = None,
) -> str:
    """Build a report while preserving separate judge and human findings."""
    application_ids = [result.question_id for result in results]
    judge_ids = [record["question_id"] for record in records]

    # Duplicate IDs could silently overwrite records when building lookups.
    if (
        len(set(application_ids)) != len(application_ids)
        or len(set(judge_ids)) != len(judge_ids)
    ):
        raise ValueError("Duplicate question IDs in report inputs")

    if set(application_ids) != set(judge_ids):
        raise ValueError("Application and judge question IDs must match")

    # Only successful judge executions contribute claim assessments.
    assessments: dict[str, JudgeAssessment] = {}
    for record in records:
        if record["status"] == "assessed":
            assessments[record["question_id"]] = (
                JudgeAssessment.model_validate(record["assessment"])
            )

    # Claim positions match those used when creating human-review-001.json.
    valid_claim_ids = {
        f"{question_id}:claim-{position}"
        for question_id, assessment in assessments.items()
        for position, claim in enumerate(assessment.claims, start=1)
        if claim.verdict != "supported"
    }

    review_by_id: dict[str, dict] = {}
    for review in reviews or []:
        review_id = review["review_id"]

        if review_id in review_by_id:
            raise ValueError(f"Duplicate review ID: {review_id}")

        if review_id not in valid_claim_ids:
            raise ValueError(
                f"Review does not match a flagged claim: {review_id}"
            )

        review_by_id[review_id] = review["human_review"]

    # Keep the original judge counts unchanged by human-review decisions.
    verdicts = Counter(
        claim.verdict
        for assessment in assessments.values()
        for claim in assessment.claims
    )
    statuses = Counter(record["status"] for record in records)
    total_claims = sum(verdicts.values())
    support_rate = (
        f"{verdicts['supported'] / total_claims:.1%}"
        if total_claims
        else "n/a"
    )

    lines = [
        "# Combined RAG evaluation report",
        "",
        "Judge findings are provisional and require human review.",
        "Claim support measures grounding in retrieved context, "
        "not overall correctness or completeness.",
        "Refusal outcomes use the application's recorded refusal flag "
        "and the dataset's question type.",
        "Refusal wording can itself be flagged by the judge; "
        "review those flags separately from substantive factual claims.",
        "",
        f"- Application results: {len(results)}",
        f"- Assessed: {statuses['assessed']}",
        f"- Skipped: {statuses['skipped']}",
        f"- Judge errors: {statuses['judge_error']}",
        f"- Claims assessed: {total_claims}",
        f"- Judge-reported claim support rate: {support_rate}",
        "",
        "| Claim verdict | Count |",
        "|---|---|",
    ]

    for verdict in (
        "supported",
        "contradicted",
        "insufficient_evidence",
        "uncertain",
    ):
        lines.append(f"| {verdict} | {verdicts[verdict]} |")

    lines.extend([
        "",
        "## Results by question",
        "",
        "| Question | Source recall | Refusal outcome | Judge status | "
        "Supported claims / total |",
        "|---|---|---|---|---|",
    ])

    by_id = {record["question_id"]: record for record in records}

    for result in results:
        record = by_id[result.question_id]
        assessment = assessments.get(result.question_id)

        recall = (
            f"{result.retrieval_recall:.0%}"
            if result.retrieval_recall is not None
            else "n/a"
        )

        # An assessment with no factual claims has no claim-support ratio.
        claims = "n/a"
        if assessment is not None and assessment.claims:
            supported = sum(
                claim.verdict == "supported"
                for claim in assessment.claims
            )
            claims = f"{supported} / {len(assessment.claims)}"

        lines.append(
            f"| {_cell(result.question_id)} | {recall} | "
            f"{_refusal_outcome(result)} | {_cell(record['status'])} | "
            f"{claims} |"
        )

    lines.extend(["", "## Claims requiring review", ""])

    for result in results:
        assessment = assessments.get(result.question_id)
        if assessment is None:
            continue

        # Preserve original positions even when supported claims are skipped.
        for position, claim in enumerate(assessment.claims, start=1):
            if claim.verdict == "supported":
                continue

            review_id = f"{result.question_id}:claim-{position}"
            evidence = ", ".join(claim.evidence_chunk_ids) or "none"

            lines.extend([
                f"### {_cell(result.question_id)} — {claim.verdict}",
                "",
                f"Review ID: {_cell(review_id)}",
                "",
                f"Question: {_cell(result.question)}",
                "",
                f"Claim: {_cell(claim.claim)}",
                "",
                f"Judge reason: {_cell(claim.reason)}",
                "",
                f"Evidence IDs: {_cell(evidence)}",
                "",
                *_human_review_lines(review_by_id.get(review_id)),
                "",
            ])

    return "\n".join(lines) + "\n"