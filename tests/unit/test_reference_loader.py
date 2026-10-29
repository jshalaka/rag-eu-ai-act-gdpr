import hashlib
import json
from pathlib import Path

import pytest

from rag_eu_law.evaluation.reference_loader import load_reference_pilot
from tests.fixtures import erasure_chunk


QUESTIONS = Path("eval/questions.yaml")


def _save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def reference_files(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Build isolated test data without changing the real pilot or corpus."""
    chunk = erasure_chunk().model_dump(mode="json")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    payload = {
        "version": "test-v1",
        "status": "draft",
        "chunks_sha256": hashlib.sha256(
            chunks_path.read_bytes()
        ).hexdigest(),
        "questions": [
            {
                "question_id": "d01",
                "question": "What does GDPR Article 17 say?",
                "required_facts": [
                    {
                        "fact_id": "d01-f1",
                        "expected_fact": "Fixture fact about erasure.",
                        "evidence_chunk_ids": [chunk["chunk_id"]],
                        "review": {
                            "status": "approved",
                            "reviewer": "Test reviewer",
                            "reason": "Fixture approval for loader testing.",
                            "reviewed_at": "2026-10-25",
                        },
                    }
                ],
                "reference_evidence": [chunk],
            }
        ],
    }

    reference_path = tmp_path / "references.json"
    _save(reference_path, payload)

    return reference_path, chunks_path, payload


def test_valid_reference_pilot_loads(reference_files) -> None:
    reference_path, chunks_path, payload = reference_files

    loaded = load_reference_pilot(
        reference_path, chunks_path, QUESTIONS
    )

    assert loaded == payload


def test_changed_corpus_is_rejected(reference_files) -> None:
    reference_path, chunks_path, _ = reference_files

    # Even a byte-level change invalidates the recorded corpus fingerprint.
    with chunks_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="corpus hash"):
        load_reference_pilot(reference_path, chunks_path, QUESTIONS)


def test_changed_evidence_text_is_rejected(reference_files) -> None:
    reference_path, chunks_path, payload = reference_files
    payload["questions"][0]["reference_evidence"][0]["text"] = (
        "Altered evidence text."
    )
    _save(reference_path, payload)

    with pytest.raises(ValueError, match="evidence differs"):
        load_reference_pilot(reference_path, chunks_path, QUESTIONS)


@pytest.mark.parametrize("duplicate_kind", ["fact", "question"])
def test_duplicate_identifiers_are_rejected(
    reference_files, duplicate_kind: str
) -> None:
    reference_path, chunks_path, payload = reference_files
    question = payload["questions"][0]

    if duplicate_kind == "fact":
        question["required_facts"].append(question["required_facts"][0])
        message = "Duplicate reference fact"
    else:
        payload["questions"].append(question)
        message = "Duplicate reference question"

    _save(reference_path, payload)

    with pytest.raises(ValueError, match=message):
        load_reference_pilot(reference_path, chunks_path, QUESTIONS)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("pending_fact", "not approved"),
        ("missing_reviewer", "requires reviewer"),
        ("invalid_date", "Invalid isoformat"),
        ("empty_fact", "Empty expected fact"),
        ("changed_question", "Question text differs"),
        ("missing_evidence", "unavailable evidence"),
    ],
)
def test_invalid_reference_details_are_rejected(
    reference_files, problem: str, message: str
) -> None:
    reference_path, chunks_path, payload = reference_files
    question = payload["questions"][0]
    fact = question["required_facts"][0]

    if problem == "pending_fact":
        fact["review"]["status"] = "pending"
    elif problem == "missing_reviewer":
        fact["review"]["reviewer"] = ""
    elif problem == "invalid_date":
        fact["review"]["reviewed_at"] = "not-a-date"
    elif problem == "empty_fact":
        fact["expected_fact"] = " "
    elif problem == "changed_question":
        question["question"] = "Different question?"
    elif problem == "missing_evidence":
        fact["evidence_chunk_ids"] = ["unknown-chunk"]

    _save(reference_path, payload)

    with pytest.raises(ValueError, match=message):
        load_reference_pilot(reference_path, chunks_path, QUESTIONS)