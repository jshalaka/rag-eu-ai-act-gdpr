"""Test provenance fingerprints using isolated temporary artifacts."""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from rag_eu_law.app.config import Settings
from rag_eu_law.domain import constants
from rag_eu_law.evaluation import provenance


@pytest.fixture
def provenance_inputs(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    vector_dir = processed_dir / constants.CHROMA_DIRNAME

    raw_dir.mkdir()
    vector_dir.mkdir(parents=True)

    # These fixtures test byte fingerprints, not PDF or index validity.
    paths = {
        "questions": tmp_path / "questions.yaml",
        "pdf": raw_dir / "example.pdf",
        "chunks": processed_dir / constants.CHUNKS_FILENAME,
        "keyword": processed_dir / constants.KEYWORD_INDEX_FILENAME,
        "vector": vector_dir / "example-index.bin",
    }

    for name, path in paths.items():
        path.write_text(f"fixture-{name}", encoding="utf-8")

    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        run_store="sqlite",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        vector_store=constants.VECTOR_STORE_CHROMA,
        top_k=5,
    )

    # Keep package metadata stable and independent of optional installations.
    monkeypatch.setattr(
        provenance,
        "_package_versions",
        lambda: {"example-package": "1.0"},
    )
    # Keep collector tests independent of a real Chroma database.
    # Real fingerprint behaviour is tested in test_index_fingerprint.py.
    monkeypatch.setattr(
        provenance,
        "capture_chroma_fingerprint",
        lambda directory, collection_name: {
            "fingerprint_version": "fixture-v1",
            "identity_sha256": "fixture-logical-index",
        },
    )
    return settings, paths


def _capture(settings, paths):
    return provenance.capture_retrieval_provenance(
        settings,
        paths["questions"],
    )


def test_unchanged_inputs_have_stable_identity(provenance_inputs) -> None:
    settings, paths = provenance_inputs

    first = _capture(settings, paths)
    second = _capture(settings, paths)

    # Capture timestamps are deliberately excluded from the identity.
    assert first["identity"] == second["identity"]
    assert first["identity_sha256"] == second["identity_sha256"]


@pytest.mark.parametrize(
    "artifact",
    ["questions", "pdf", "chunks", "keyword", "vector"],
)
def test_changed_artifact_changes_identity(
    provenance_inputs, artifact: str
) -> None:
    settings, paths = provenance_inputs
    previous = _capture(settings, paths)

    paths[artifact].write_text("changed content", encoding="utf-8")
    current = _capture(settings, paths)

    assert current["identity_sha256"] != previous["identity_sha256"]


def test_changed_retrieval_setting_changes_identity(
    provenance_inputs,
) -> None:
    settings, paths = provenance_inputs
    previous = _capture(settings, paths)

    changed = settings.model_copy(update={"top_k": 10})
    current = _capture(changed, paths)

    assert current["identity_sha256"] != previous["identity_sha256"]


def test_credentials_are_excluded_from_identity(provenance_inputs) -> None:
    settings, paths = provenance_inputs

    first_settings = settings.model_copy(
        update={"openai_api_key": SecretStr("fixture-secret-one")}
    )
    second_settings = settings.model_copy(
        update={"openai_api_key": SecretStr("fixture-secret-two")}
    )

    first = _capture(first_settings, paths)
    second = _capture(second_settings, paths)

    assert first["identity_sha256"] == second["identity_sha256"]
    serialized = json.dumps([first, second])
    assert "fixture-secret-one" not in serialized
    assert "fixture-secret-two" not in serialized


def test_unsupported_backend_is_rejected(provenance_inputs) -> None:
    settings, paths = provenance_inputs
    changed = settings.model_copy(
        update={"vector_store": constants.VECTOR_STORE_MEMORY}
    )

    with pytest.raises(ValueError, match="requires Chroma"):
        _capture(changed, paths)


def test_missing_index_directory_is_rejected(provenance_inputs) -> None:
    settings, paths = provenance_inputs
    paths["vector"].unlink()
    paths["vector"].parent.rmdir()

    with pytest.raises(ValueError, match="directory is missing"):
        _capture(settings, paths)

def test_sqlite_byte_changes_are_recorded_as_diagnostics(
    provenance_inputs,
) -> None:
    settings, paths = provenance_inputs
    sqlite_path = paths["vector"].parent / "chroma.sqlite3"
    sqlite_path.write_bytes(b"database-state-one")

    first = _capture(settings, paths)

    sqlite_path.write_bytes(b"database-state-two")
    second = _capture(settings, paths)

    # The mocked logical fingerprint and search artifacts are unchanged.
    assert first["identity_sha256"] == second["identity_sha256"]

    # Physical changes must remain visible in the recorded evidence.
    first_files = first["diagnostics"]["vector_index_physical_files"]
    second_files = second["diagnostics"]["vector_index_physical_files"]
    assert first_files["chroma.sqlite3"] != second_files["chroma.sqlite3"]


def test_logical_index_change_changes_compatibility_identity(
    provenance_inputs, monkeypatch
) -> None:
    settings, paths = provenance_inputs
    first = _capture(settings, paths)

    monkeypatch.setattr(
        provenance,
        "capture_chroma_fingerprint",
        lambda directory, collection_name: {
            "fingerprint_version": "fixture-v1",
            "identity_sha256": "changed-logical-index",
        },
    )
    second = _capture(settings, paths)

    assert first["identity_sha256"] != second["identity_sha256"]


def test_search_artifact_changes_remain_compatibility_inputs(
    provenance_inputs,
) -> None:
    settings, paths = provenance_inputs
    first = _capture(settings, paths)

    paths["vector"].write_bytes(b"changed-search-index")
    second = _capture(settings, paths)

    assert first["identity_sha256"] != second["identity_sha256"]
    assert (
        first["identity"]["vector_index"]["search_index_files"]
        != second["identity"]["vector_index"]["search_index_files"]
    )