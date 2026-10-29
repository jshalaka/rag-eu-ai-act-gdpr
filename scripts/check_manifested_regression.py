"""Compare retrieval runs after verifying version-2 provenance."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_eu_law.evaluation.models import QuestionResult
from rag_eu_law.evaluation.regression import compare_retrieval_runs


IDENTITY_FIELDS = {
    "questions_sha256",
    "raw_pdfs",
    "chunks_sha256",
    "keyword_index_sha256",
    "vector_index",
    "retrieval_settings",
    "application_source_files",
    "environment",
}


def _verify_snapshot(snapshot: dict) -> dict:
    """Check that the recorded digest matches the compatibility identity."""
    if snapshot.get("schema_version") != "retrieval-provenance-v2":
        raise ValueError("This command requires version-2 provenance")

    identity = snapshot["identity"]
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ValueError("Unexpected provenance identity fields")

    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != snapshot["identity_sha256"]:
        raise ValueError("Provenance identity digest mismatch")

    return identity


def load_verified_run(
    results_path: Path,
) -> tuple[list[QuestionResult], dict, dict]:
    """Check run completion, captured input stability, and result integrity."""
    manifest_path = results_path.parent / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("manifest_version") != "retrieval-run-v1":
        raise ValueError("Unsupported run manifest version")
    if manifest.get("mode") != "retrieval_only":
        raise ValueError("Expected a retrieval-only run")
    if manifest.get("status") != "completed":
        raise ValueError("Run did not complete successfully")
    if manifest.get("inputs_stable") is not True:
        raise ValueError("Run inputs were not stable")
    if manifest.get("application_errors") != 0:
        raise ValueError("Run recorded application errors")

    before = _verify_snapshot(manifest["before"])
    after = _verify_snapshot(manifest["after"])

    # Independently verify the recorded stability flag.
    if before != after:
        raise ValueError("Before/after provenance identities differ")

    raw_results = results_path.read_bytes()
    if hashlib.sha256(raw_results).hexdigest() != manifest["results_sha256"]:
        raise ValueError("Results file does not match its manifest hash")

    payload = json.loads(raw_results)
    results = [
        QuestionResult.model_validate(item)
        for item in payload["results"]
    ]

    return results, before, manifest


def _physical_differences(baseline: dict, candidate: dict) -> list[str]:
    """Report physical changes separately from compatibility decisions."""
    previous = baseline["before"]["diagnostics"][
        "vector_index_physical_files"
    ]
    current = candidate["before"]["diagnostics"][
        "vector_index_physical_files"
    ]

    return sorted(
        name
        for name in set(previous) | set(current)
        if previous.get(name) != current.get(name)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")

        baseline, baseline_identity, baseline_manifest = load_verified_run(
            args.baseline
        )
        candidate, candidate_identity, candidate_manifest = load_verified_run(
            args.candidate
        )

        differences = sorted(
            field
            for field in IDENTITY_FIELDS
            if baseline_identity[field] != candidate_identity[field]
        )

        physical_changes = _physical_differences(
            baseline_manifest,
            candidate_manifest,
        )

        if differences:
            # Different inputs prevent a valid strict non-decline comparison.
            report = {
                "passed": False,
                "comparison_performed": False,
                "provenance_compatible": False,
                "incompatible_fields": differences,
            }
            exit_code = 2
        else:
            report = compare_retrieval_runs(baseline, candidate)
            report.update(
                comparison_performed=True,
                provenance_compatible=True,
                incompatible_fields=[],
            )
            exit_code = 0 if report["passed"] else 1

        report.update(
            baseline_file=args.baseline.as_posix(),
            candidate_file=args.candidate.as_posix(),
            diagnostic_physical_file_differences=physical_changes,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"INPUT ERROR: {error}")
        return 2

    print("Provenance compatible:", report["provenance_compatible"])
    print("Physical file differences (diagnostic):", physical_changes)

    if differences:
        print("Incompatible fields:", ", ".join(differences))
        print("Score comparison was not performed.")
    else:
        for name, passed in report["checks"].items():
            print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print("Report saved:", args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
