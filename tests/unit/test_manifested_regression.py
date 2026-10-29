"""Verify strict comparison using isolated synthetic run files."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def command():
    """Load the script without executing its command-line entry point."""
    path = Path("scripts/check_manifested_regression.py")
    spec = importlib.util.spec_from_file_location("manifested_command", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(identity: dict) -> dict:
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return {
        "schema_version": "retrieval-provenance-v2",
        "identity": deepcopy(identity),
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "diagnostics": {
            "vector_index_physical_files": {
                "chroma.sqlite3": "physical-state-one"
            }
        },
    }


def _write_run(folder: Path) -> Path:
    """Build a verifier fixture; this is not an executed evaluation."""
    folder.mkdir()
    results_path = folder / "results.json"

    _write_json(results_path, {
        "results": [
            {
                "question_id": "q1",
                "type": "concept",
                "question": "Example question?",
                "retrieved": ["chunk-1"],
                "retrieval_hit": True,
                "retrieval_recall": 1.0,
                "missing_sources": [],
            }
        ]
    })

    identity = {
        "questions_sha256": "fixture-questions",
        "raw_pdfs": {"example.pdf": "fixture-pdf"},
        "chunks_sha256": "fixture-chunks",
        "keyword_index_sha256": "fixture-keyword",
        "vector_index": {"logical": "fixture-index"},
        "retrieval_settings": {"top_k": 5},
        "application_source_files": {"example.py": "fixture-code"},
        "environment": {"python": "fixture-version"},
    }

    _write_json(folder / "provenance.json", {
        "manifest_version": "retrieval-run-v1",
        "mode": "retrieval_only",
        "status": "completed",
        "inputs_stable": True,
        "application_errors": 0,
        "before": _snapshot(identity),
        "after": _snapshot(identity),
        "results_sha256": hashlib.sha256(
            results_path.read_bytes()
        ).hexdigest(),
    })

    return results_path


@pytest.fixture
def runs(tmp_path):
    return {
        "baseline": _write_run(tmp_path / "baseline"),
        "candidate": _write_run(tmp_path / "candidate"),
        "output": tmp_path / "comparison.json",
    }


def _invoke(command, runs) -> int:
    return command.main([
        "--baseline", str(runs["baseline"]),
        "--candidate", str(runs["candidate"]),
        "--output", str(runs["output"]),
    ])


def _manifest(runs) -> tuple[Path, dict]:
    path = runs["candidate"].parent / "provenance.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_compatible_runs_allow_sqlite_diagnostic_difference(
    command, runs
) -> None:
    path, manifest = _manifest(runs)
    for stage in ("before", "after"):
        manifest[stage]["diagnostics"]["vector_index_physical_files"][
            "chroma.sqlite3"
        ] = "physical-state-two"
    _write_json(path, manifest)

    assert _invoke(command, runs) == 0

    report = json.loads(runs["output"].read_text(encoding="utf-8"))
    assert report["provenance_compatible"] is True
    assert report["passed"] is True
    assert report["diagnostic_physical_file_differences"] == [
        "chroma.sqlite3"
    ]


def test_changed_results_without_updated_hash_are_rejected(
    command, runs
) -> None:
    # A byte change alone must invalidate the recorded results digest.
    with runs["candidate"].open("ab") as handle:
        handle.write(b"\n")

    assert _invoke(command, runs) == 2
    assert not runs["output"].exists()


def test_altered_snapshot_without_updated_digest_is_rejected(
    command, runs
) -> None:
    path, manifest = _manifest(runs)
    manifest["before"]["identity"]["retrieval_settings"]["top_k"] = 10
    _write_json(path, manifest)

    assert _invoke(command, runs) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("inputs_stable", False),
        ("application_errors", 1),
    ],
)
def test_unsuitable_run_is_rejected(command, runs, field, value) -> None:
    path, manifest = _manifest(runs)
    manifest[field] = value
    _write_json(path, manifest)

    assert _invoke(command, runs) == 2


def test_valid_but_different_settings_prevent_comparison(
    command, runs
) -> None:
    path, manifest = _manifest(runs)
    changed = deepcopy(manifest["before"]["identity"])
    changed["retrieval_settings"]["top_k"] = 10

    # Recompute both snapshots: integrity passes, compatibility must fail.
    manifest["before"] = _snapshot(changed)
    manifest["after"] = _snapshot(changed)
    _write_json(path, manifest)

    assert _invoke(command, runs) == 2

    report = json.loads(runs["output"].read_text(encoding="utf-8"))
    assert report["comparison_performed"] is False
    assert report["incompatible_fields"] == ["retrieval_settings"]


def test_compatible_run_with_retrieval_decline_returns_one(
    command, runs
) -> None:
    results_path = runs["candidate"]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["results"][0].update(
        retrieved=[],
        retrieval_hit=False,
        retrieval_recall=0.0,
        missing_sources=[{"regulation": "GDPR", "article": "17"}],
    )
    _write_json(results_path, payload)

    # This models a different recorded outcome with its matching digest.
    path, manifest = _manifest(runs)
    manifest["results_sha256"] = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    _write_json(path, manifest)

    assert _invoke(command, runs) == 1

    report = json.loads(runs["output"].read_text(encoding="utf-8"))
    assert report["provenance_compatible"] is True
    assert report["comparison_performed"] is True
    assert report["passed"] is False
    assert report["question_regressions"][0]["question_id"] == "q1"