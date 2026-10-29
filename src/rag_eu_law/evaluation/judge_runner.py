from __future__ import annotations

from rag_eu_law.domain.exceptions import RagError
from rag_eu_law.evaluation.judge import LLMJudge, RUBRIC_VERSION
from rag_eu_law.evaluation.models import QuestionResult


def judge_results(
    results: list[QuestionResult],
    judge: LLMJudge,
    *,
    judge_model: str,
) -> list[dict]:
    """Judge saved application results without rerunning the application."""
    records = []

    for result in results:
        record = {
            "question_id": result.question_id,
            "rubric_version": RUBRIC_VERSION,
            "judge_model": judge_model,
            "status": None,
            "skip_reason": None,
            "assessment": None,
            "judge_error": None,
        }

        if result.error is not None:
            record.update(
                status="skipped",
                skip_reason="application_error",
            )
        elif result.answer is None or not result.answer.strip():
            record.update(
                status="skipped",
                skip_reason="missing_answer",
            )
        elif result.retrieved_context is None:
            record.update(
                status="skipped",
                skip_reason="missing_context",
            )
        else:
            try:
                assessment = judge.assess(result)
            except RagError as error:
                record.update(
                    status="judge_error",
                    judge_error=str(error),
                )
            else:
                record.update(
                    status="assessed",
                    assessment=assessment.model_dump(mode="json"),
                )

        records.append(record)

    return records