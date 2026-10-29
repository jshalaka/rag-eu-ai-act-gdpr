import pytest

from rag_eu_law.domain.exceptions import EvaluationError
from rag_eu_law.evaluation.judge_models import JudgeAssessment
from rag_eu_law.evaluation.judge_runner import judge_results
from rag_eu_law.evaluation.models import QuestionResult, QuestionType


class StubJudge:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls = []

    def assess(self, result: QuestionResult) -> JudgeAssessment:
        self.calls.append(result.question_id)

        if result.question_id == self.fail_on:
            raise EvaluationError("Invalid judge response")

        return JudgeAssessment(
            answer_behavior="insufficient_evidence",
            relevance="relevant",
            claims=[],
            reason="The answer acknowledges insufficient evidence.",
        )


def _result(qid: str = "q1", **updates) -> QuestionResult:
    fields = {
        "question_id": qid,
        "type": QuestionType.CONCEPT,
        "question": "What evidence is available?",
        "retrieved": [],
        "retrieved_context": [],
        "answer": "The supplied context does not support an answer.",
    }
    fields.update(updates)
    return QuestionResult.model_validate(fields)


def test_successful_assessment_records_metadata() -> None:
    judge = StubJudge()

    records = judge_results(
        [_result()], judge, judge_model="test-model"
    )

    assert len(records) == 1
    record = records[0]
    assert record["question_id"] == "q1"
    assert record["judge_model"] == "test-model"
    assert record["rubric_version"] == "grounding-v1"
    assert record["status"] == "assessed"
    assert record["assessment"]["answer_behavior"] == "insufficient_evidence"
    assert record["judge_error"] is None
    assert record["skip_reason"] is None
    assert judge.calls == ["q1"]


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"error": "Generation failed"}, "application_error"),
        ({"answer": None}, "missing_answer"),
        ({"answer": "   "}, "missing_answer"),
        ({"retrieved_context": None}, "missing_context"),
    ],
)
def test_unsuitable_results_are_skipped_without_judge_calls(
    updates: dict, reason: str
) -> None:
    judge = StubJudge()

    records = judge_results(
        [_result(**updates)], judge, judge_model="test-model"
    )

    assert records[0]["status"] == "skipped"
    assert records[0]["skip_reason"] == reason
    assert records[0]["assessment"] is None
    assert records[0]["judge_error"] is None
    assert judge.calls == []


def test_judge_failure_does_not_stop_later_results() -> None:
    judge = StubJudge(fail_on="bad")

    records = judge_results(
        [_result("bad"), _result("good")],
        judge,
        judge_model="test-model",
    )

    assert [record["question_id"] for record in records] == ["bad", "good"]
    assert [record["status"] for record in records] == [
        "judge_error", "assessed"
    ]
    assert records[0]["judge_error"] == "Invalid judge response"
    assert records[0]["assessment"] is None
    assert judge.calls == ["bad", "good"]


def test_empty_batch_makes_no_judge_calls() -> None:
    judge = StubJudge()

    assert judge_results([], judge, judge_model="test-model") == []
    assert judge.calls == []