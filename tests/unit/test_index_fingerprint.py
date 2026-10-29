"""Test logical fingerprints independently of a real Chroma database."""

from copy import deepcopy

import pytest

from rag_eu_law.evaluation.index_fingerprint import (
    fingerprint_collection,
    fingerprint_records,
)


def _records() -> list[dict]:
    return [
        {
            "id": "a",
            "document": "First document.",
            "metadata": {"article": "17"},
            "embedding": [0.1, 0.2],
        },
        {
            "id": "b",
            "document": "Second document.",
            "metadata": {"article": "6"},
            "embedding": [0.3, 0.4],
        },
    ]


def _fingerprint(records, **updates) -> dict:
    options = {
        "collection_name": "test_collection",
        "configuration": {"hnsw": {"space": "cosine"}},
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    options.update(updates)
    return fingerprint_records(records, **options)


def test_record_order_does_not_change_fingerprint() -> None:
    records = _records()

    assert _fingerprint(records) == _fingerprint(list(reversed(records)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "changed-id"),
        ("document", "Changed document."),
        ("metadata", {"article": "9"}),
        ("embedding", [0.1, 0.25]),
    ],
)
def test_changed_record_changes_fingerprint(field, value) -> None:
    original = _records()
    changed = deepcopy(original)
    changed[0][field] = value

    assert (
        _fingerprint(original)["identity_sha256"]
        != _fingerprint(changed)["identity_sha256"]
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"configuration": {"hnsw": {"space": "l2"}}},
        {"collection_metadata": {"different": "metadata"}},
        {"collection_name": "another_collection"},
    ],
)
def test_changed_collection_details_change_fingerprint(updates) -> None:
    assert (
        _fingerprint(_records())["identity_sha256"]
        != _fingerprint(_records(), **updates)["identity_sha256"]
    )


def test_duplicate_ids_are_rejected() -> None:
    records = _records()
    records[1]["id"] = records[0]["id"]

    with pytest.raises(ValueError, match="Duplicate vector record"):
        _fingerprint(records)


@pytest.mark.parametrize(
    "embedding",
    [[], [float("nan"), 0.2], [float("inf"), 0.2]],
)
def test_invalid_vectors_are_rejected(embedding) -> None:
    records = _records()
    records[0]["embedding"] = embedding

    with pytest.raises(ValueError, match="Invalid embedding"):
        _fingerprint(records)


def test_inconsistent_dimensions_are_rejected() -> None:
    records = _records()
    records[1]["embedding"] = [0.3]

    with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
        _fingerprint(records)


def test_empty_collection_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty vector collection"):
        _fingerprint([])


class FakeCollection:
    name = "test_collection"
    configuration_json = {"hnsw": {"space": "cosine"}}
    metadata = {"hnsw:space": "cosine"}

    def __init__(self) -> None:
        self.records = _records()
        self.requested_fields = []

    def count(self) -> int:
        return len(self.records)

    def get(self, *, limit, offset, include):
        self.requested_fields.append(include)
        selected = self.records[offset:offset + limit]
        return {
            "ids": [record["id"] for record in selected],
            "documents": [record["document"] for record in selected],
            "metadatas": [record["metadata"] for record in selected],
            "embeddings": [record["embedding"] for record in selected],
        }


def test_paginated_collection_matches_direct_fingerprint() -> None:
    collection = FakeCollection()

    actual = fingerprint_collection(collection, batch_size=1)

    assert actual == _fingerprint(_records())
    assert len(collection.requested_fields) == 2
    assert all(
        fields == ["documents", "metadatas", "embeddings"]
        for fields in collection.requested_fields
    )


def test_changed_collection_count_is_rejected() -> None:
    class ChangingCollection(FakeCollection):
        def __init__(self):
            super().__init__()
            self.count_calls = 0

        def count(self):
            self.count_calls += 1
            return 2 if self.count_calls == 1 else 3

    with pytest.raises(ValueError, match="count changed"):
        fingerprint_collection(ChangingCollection())


def test_incomplete_collection_response_is_rejected() -> None:
    class IncompleteCollection(FakeCollection):
        def get(self, **kwargs):
            response = super().get(**kwargs)
            response["embeddings"] = None
            return response

    with pytest.raises(ValueError, match="incomplete record fields"):
        fingerprint_collection(IncompleteCollection())