"""Load and validate the evaluation question set."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from rag_eu_law.domain.exceptions import EvaluationError
from rag_eu_law.evaluation.models import EvalQuestion

MIN_QUESTIONS = 20


def load_questions(path: Path) -> list[EvalQuestion]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationError(f"Cannot read questions from {path}: {error}") from error
    if not isinstance(raw, list):
        raise EvaluationError(f"{path} must contain a list of questions")
    try:
        questions = [EvalQuestion.model_validate(item) for item in raw]
    except ValidationError as error:
        raise EvaluationError(f"Invalid question in {path}: {error}") from error
    _check_set(questions, path)
    return questions


def _check_set(questions: list[EvalQuestion], path: Path) -> None:
    ids = [question.id for question in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise EvaluationError(f"{path}: duplicate question ids {sorted(duplicates)}")
    if len(questions) < MIN_QUESTIONS:
        raise EvaluationError(
            f"{path}: need at least {MIN_QUESTIONS} questions, got {len(questions)}"
        )
