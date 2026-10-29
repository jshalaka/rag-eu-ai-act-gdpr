import json

import pytest

from rag_eu_law.domain.exceptions import EvaluationError
from rag_eu_law.evaluation.models import QuestionResult, QuestionType
from rag_eu_law.evaluation.reference_judge import ReferenceCoverageJudge


class FakeLLM:
    """Return a controlled response instead of calling a real model."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


def _result(**updates) -> QuestionResult:
    values = {
        "question_id": "d01",
        "type": QuestionType.DIRECT_LOOKUP,
        "question": "What does GDPR Article 17 say?",
        "retrieved": [],
        "answer": "Erasure is conditional.",
    }
    values.update(updates)
    return QuestionResult.model_validate(values)


def _reference() -> dict:
    # These fixtures test adapter validation, not source or legal correctness.
    return {
        "question_id": "d01",
        "question": "What does GDPR Article 17 say?",
        "required_facts": [
            {
                "fact_id": "f1",
                "expected_fact": "Erasure is conditional.",
                "evidence_chunk_ids": ["example-1"],
                "review": {"status": "approved"},
            },
            {
                "fact_id": "f2",
                "expected_fact": "Exceptions apply.",
                "evidence_chunk_ids": ["example-2"],
                "review": {"status": "approved"},
            },
        ],
        "reference_evidence": [],
    }


def _assessment() -> dict:
    return {
        "facts": [
            {
                "fact_id": "f1",
                "verdict": "covered",
                "answer_quotes": ["Erasure is conditional."],
                "reason": "The answer states the condition.",
            },
            {
                "fact_id": "f2",
                "verdict": "missing",
                "answer_quotes": [],
                "reason": "The answer does not mention exceptions.",
            },
        ],
        "reason": "One required fact is covered; one is missing.",
    }


def test_valid_reference_assessment_is_accepted() -> None:
    llm = FakeLLM(json.dumps(_assessment()))

    assessment = ReferenceCoverageJudge(llm).assess(
        _result(), _reference()
    )

    assert [fact.verdict for fact in assessment.facts] == [
        "covered", "missing"
    ]
    assert llm.calls == 1


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(EvaluationError, match="invalid assessment JSON"):
        ReferenceCoverageJudge(FakeLLM("not JSON")).assess(
            _result(), _reference()
        )


@pytest.mark.parametrize("problem", ["omitted", "invented", "duplicate"])
def test_fact_ids_must_match_exactly(problem: str) -> None:
    payload = _assessment()

    if problem == "omitted":
        payload["facts"].pop()
    elif problem == "invented":
        payload["facts"][1]["fact_id"] = "unknown"
    else:
        payload["facts"][1]["fact_id"] = "f1"

    with pytest.raises(EvaluationError, match="exactly once"):
        ReferenceCoverageJudge(FakeLLM(json.dumps(payload))).assess(
            _result(), _reference()
        )


def test_invented_answer_quote_is_rejected() -> None:
    payload = _assessment()
    payload["facts"][0]["answer_quotes"] = ["This sentence was invented."]

    with pytest.raises(EvaluationError, match="absent from the answer"):
        ReferenceCoverageJudge(FakeLLM(json.dumps(payload))).assess(
            _result(), _reference()
        )


@pytest.mark.parametrize(
    "problem",
    ["covered_without_quote", "missing_with_quote", "empty_quote"],
)
def test_invalid_quote_requirements_are_rejected(problem: str) -> None:
    payload = _assessment()

    if problem == "covered_without_quote":
        payload["facts"][0]["answer_quotes"] = []
    elif problem == "missing_with_quote":
        payload["facts"][1]["answer_quotes"] = ["Erasure is conditional."]
    else:
        payload["facts"][0]["answer_quotes"] = [" "]

    with pytest.raises(EvaluationError, match="invalid assessment JSON"):
        ReferenceCoverageJudge(FakeLLM(json.dumps(payload))).assess(
            _result(), _reference()
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"answer": None},
        {"answer": " "},
        {"error": "Application failed"},
    ],
)
def test_unusable_answers_make_no_model_call(updates: dict) -> None:
    llm = FakeLLM("{}")

    with pytest.raises(EvaluationError):
        ReferenceCoverageJudge(llm).assess(
            _result(**updates), _reference()
        )

    assert llm.calls == 0


def test_question_mismatch_makes_no_model_call() -> None:
    llm = FakeLLM("{}")
    reference = _reference()
    reference["question"] = "Different question?"

    with pytest.raises(EvaluationError, match="do not match"):
        ReferenceCoverageJudge(llm).assess(_result(), reference)

    assert llm.calls == 0


def test_unapproved_fact_makes_no_model_call() -> None:
    llm = FakeLLM("{}")
    reference = _reference()
    reference["required_facts"][0]["review"]["status"] = "pending"

    with pytest.raises(EvaluationError, match="must be approved"):
        ReferenceCoverageJudge(llm).assess(_result(), reference)

    assert llm.calls == 0