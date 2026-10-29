import json

import pytest

from rag_eu_law.domain.exceptions import EvaluationError
from rag_eu_law.evaluation.judge import LLMJudge
from rag_eu_law.evaluation.models import QuestionResult, QuestionType
from tests.fixtures import erasure_chunk


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


def _result() -> QuestionResult:
    chunk = erasure_chunk()
    return QuestionResult(
        question_id="judge-001",
        type=QuestionType.DIRECT_LOOKUP,
        question="What does GDPR Article 17 cover?",
        retrieved=[chunk.chunk_id],
        retrieved_context=[chunk],
        answer="GDPR Article 17 covers the right to erasure.",
        citations=["[GDPR Art. 17, p. 61]"],
        refused=False,
    )


def _assessment(chunk_id: str) -> dict:
    return {
        "answer_behavior": "answered",
        "relevance": "relevant",
        "claims": [
            {
                "claim": "GDPR Article 17 covers the right to erasure.",
                "verdict": "supported",
                "evidence_chunk_ids": [chunk_id],
                "reason": "The supplied chunk describes the right to erasure.",
            }
        ],
        "reason": "The answer addresses the question using supplied evidence.",
    }


def test_judge_accepts_valid_assessment() -> None:
    result = _result()
    response = _assessment(result.retrieved[0])
    llm = FakeLLM(json.dumps(response))

    assessment = LLMJudge(llm).assess(result)

    assert assessment.answer_behavior == "answered"
    assert assessment.claims[0].verdict == "supported"
    assert assessment.claims[0].evidence_chunk_ids == result.retrieved
    assert llm.calls == 1


def test_judge_rejects_invalid_json() -> None:
    llm = FakeLLM("This is not JSON.")

    with pytest.raises(EvaluationError, match="invalid assessment JSON"):
        LLMJudge(llm).assess(_result())


def test_judge_rejects_unknown_evidence_ids() -> None:
    llm = FakeLLM(json.dumps(_assessment("invented-chunk-id")))

    with pytest.raises(EvaluationError, match="unknown chunks"):
        LLMJudge(llm).assess(_result())


def test_judge_rejects_supported_claim_without_evidence() -> None:
    response = _assessment(_result().retrieved[0])
    response["claims"][0]["evidence_chunk_ids"] = []
    llm = FakeLLM(json.dumps(response))

    with pytest.raises(EvaluationError, match="invalid assessment JSON"):
        LLMJudge(llm).assess(_result())


@pytest.mark.parametrize(
    "updates",
    [
        {"answer": None},
        {"retrieved_context": None},
        {"error": "Generation failed"},
    ],
)
def test_judge_skips_unjudgeable_results_without_model_call(
    updates: dict,
) -> None:
    result = _result().model_copy(update=updates)
    llm = FakeLLM("{}")

    with pytest.raises(EvaluationError):
        LLMJudge(llm).assess(result)

    assert llm.calls == 0
