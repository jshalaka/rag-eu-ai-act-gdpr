"""Fingerprint stored vector records and collection configuration."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _canonical_bytes(value: Any) -> bytes:
    """Create a deterministic JSON representation without rounding vectors."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def fingerprint_records(
    records: list[dict],
    *,
    collection_name: str,
    configuration: dict,
    collection_metadata: dict | None,
) -> dict:
    """Fingerprint content independently of record retrieval order."""
    if not records:
        raise ValueError("Cannot fingerprint an empty vector collection")

    seen_ids = set()
    normalized = []
    dimensions = set()

    for record in records:
        record_id = record["id"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Vector record requires a nonempty ID")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate vector record ID: {record_id}")
        seen_ids.add(record_id)

        if not isinstance(record["document"], str):
            raise ValueError(f"Missing document text: {record_id}")
        if not isinstance(record["metadata"], dict):
            raise ValueError(f"Missing record metadata: {record_id}")

        vector = [float(value) for value in record["embedding"]]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid embedding: {record_id}")

        dimensions.add(len(vector))
        normalized.append({
            "id": record_id,
            "document": record["document"],
            "metadata": record["metadata"],
            "embedding": vector,
        })

    if len(dimensions) != 1:
        raise ValueError("Inconsistent embedding dimensions")

    # Sorting removes API pagination/order differences from the fingerprint.
    normalized.sort(key=lambda record: record["id"])

    result = {
        "fingerprint_version": "chroma-content-v1",
        "collection_name": collection_name,
        "record_count": len(normalized),
        "embedding_dimensions": next(iter(dimensions)),
        "records_sha256": _digest(normalized),
        "configuration_sha256": _digest(configuration),
        "collection_metadata_sha256": _digest(collection_metadata),
    }
    result["identity_sha256"] = _digest(result)

    return result


def fingerprint_collection(
    collection: Any,
    *,
    batch_size: int = 256,
) -> dict:
    """Read an existing collection; concurrent writers must be stopped."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    count_before = collection.count()
    records = []

    for offset in range(0, count_before, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = batch["ids"]
        documents = batch["documents"]
        metadatas = batch["metadatas"]
        embeddings = batch["embeddings"]

        if any(value is None for value in (documents, metadatas, embeddings)):
            raise ValueError("Collection returned incomplete record fields")

        if not (
            len(ids) == len(documents) == len(metadatas) == len(embeddings)
        ):
            raise ValueError("Collection returned mismatched field lengths")

        for record_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings, strict=True
        ):
            records.append({
                "id": record_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            })

    if collection.count() != count_before or len(records) != count_before:
        raise ValueError("Collection count changed or records were omitted")

    configuration = collection.configuration_json
    if not isinstance(configuration, dict):
        raise ValueError("Collection configuration is unavailable")

    return fingerprint_records(
        records,
        collection_name=collection.name,
        configuration=configuration,
        collection_metadata=collection.metadata,
    )


def capture_chroma_fingerprint(
    directory: Path,
    collection_name: str,
) -> dict:
    """Open an existing collection without creating or inserting records."""
    import chromadb

    if not (directory / "chroma.sqlite3").is_file():
        raise ValueError(f"Existing Chroma database not found: {directory}")

    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    return fingerprint_collection(collection)