"""Validate reviewed reference facts against the question set and corpus."""

import hashlib
import json
from datetime import date
from pathlib import Path

from rag_eu_law.domain.models import Chunk
from rag_eu_law.evaluation.loader import load_questions


def _unique_lookup(items: list[dict], key: str, label: str) -> dict:
    """Reject duplicates instead of silently overwriting evidence."""
    lookup = {}

    for item in items:
        identifier = item[key]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Empty {label} identifier")

        if identifier in lookup:
            raise ValueError(f"Duplicate {label}: {identifier}")

        lookup[identifier] = item

    return lookup


def load_reference_pilot(
    reference_path: Path,
    chunks_path: Path,
    questions_path: Path,
) -> dict:
    """Load usable reference facts; do not assess their semantic validity."""
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    chunk_bytes = chunks_path.read_bytes()

    # The reference evidence must belong to the current corpus snapshot.
    actual_hash = hashlib.sha256(chunk_bytes).hexdigest()
    if payload["chunks_sha256"] != actual_hash:
        raise ValueError("Reference corpus hash does not match current chunks")

    corpus_records = [
        json.loads(line)
        for line in chunk_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    corpus = _unique_lookup(corpus_records, "chunk_id", "corpus chunk")

    dataset = {
        question.id: question
        for question in load_questions(questions_path)
    }

    reference_questions = payload["questions"]
    if not reference_questions:
        raise ValueError("Reference pilot contains no questions")

    _unique_lookup(reference_questions, "question_id", "reference question")
    seen_fact_ids = set()

    for question in reference_questions:
        question_id = question["question_id"]

        if question_id not in dataset:
            raise ValueError(f"Unknown reference question: {question_id}")

        if question["question"] != dataset[question_id].question:
            raise ValueError(f"Question text differs: {question_id}")

        facts = question["required_facts"]
        if not facts:
            raise ValueError(f"No required facts: {question_id}")

        evidence = _unique_lookup(
            question["reference_evidence"],
            "chunk_id",
            "reference evidence",
        )

        # Validate the stored evidence against actual corpus content,
        # rather than accepting a matching chunk ID alone.
        for chunk_id, stored_chunk in evidence.items():
            if chunk_id not in corpus:
                raise ValueError(f"Unknown evidence chunk: {chunk_id}")

            stored = Chunk.model_validate(stored_chunk)
            actual = Chunk.model_validate(corpus[chunk_id])

            if stored != actual:
                raise ValueError(f"Reference evidence differs: {chunk_id}")

        for fact in facts:
            fact_id = fact["fact_id"]
            if not isinstance(fact_id, str) or not fact_id.strip():
                raise ValueError("Fact ID must be nonempty")

            if fact_id in seen_fact_ids:
                raise ValueError(f"Duplicate reference fact: {fact_id}")
            seen_fact_ids.add(fact_id)

            expected = fact["expected_fact"]
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"Empty expected fact: {fact_id}")

            evidence_ids = fact["evidence_chunk_ids"]
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"Fact requires evidence: {fact_id}")

            for chunk_id in evidence_ids:
                if chunk_id not in evidence:
                    raise ValueError(
                        f"Fact references unavailable evidence: {fact_id}"
                    )

            review = fact["review"]
            if review["status"] != "approved":
                raise ValueError(f"Reference fact is not approved: {fact_id}")

            for field in ("reviewer", "reason", "reviewed_at"):
                value = review.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Approved fact requires {field}: {fact_id}"
                    )

            # Reject invalid review dates while preserving their original value.
            date.fromisoformat(review["reviewed_at"])

    # The file remains unchanged; structural validation is not legal validation.
    return payload