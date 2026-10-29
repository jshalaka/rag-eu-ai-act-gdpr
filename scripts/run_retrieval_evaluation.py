"""Run retrieval evaluation with before/after provenance snapshots."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.app.logging_config import configure_logging
from rag_eu_law.domain import constants
from rag_eu_law.evaluation.loader import load_questions
from rag_eu_law.evaluation.metrics import summarize
from rag_eu_law.evaluation.provenance import capture_retrieval_provenance
from rag_eu_law.evaluation.report import summary_lines, write_report
from rag_eu_law.evaluation.runner import EvalRunner


def write_manifest(path: Path, payload: dict) -> None:
    """Replace the manifest atomically so partial JSON is not retained."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(constants.DEFAULT_QUESTIONS_PATH),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Preserve prior evaluation evidence.
    if args.output_dir.exists():
        raise SystemExit("Choose a new output directory.")

    settings = Settings()
    configure_logging(settings.log_level)

    container = Container(settings)

    # Initialize the retriever before capturing the execution baseline.
    # Initialization itself may open or maintain the persisted index.
    retriever = container.retriever

    before = capture_retrieval_provenance(settings, args.questions)
    questions = load_questions(args.questions)

    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "provenance.json"

    manifest = {
        "manifest_version": "retrieval-run-v1",
        "mode": "retrieval_only",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "before": before,
        "after": None,
        "inputs_stable": None,
        "application_errors": None,
        "results_sha256": None,
    }
    write_manifest(manifest_path, manifest)

    try:
        # Use the application's existing runner; no answer generator is supplied.
        results = EvalRunner(
            retriever,
            None,
            settings.top_k,
        ).run(questions)

        summary = summarize(results)
        report_path = write_report(
            results,
            summary,
            args.output_dir,
            settings.top_k,
        )

        after = capture_retrieval_provenance(settings, args.questions)
        stable = before["identity_sha256"] == after["identity_sha256"]
        errors = sum(result.error is not None for result in results)

        results_path = args.output_dir / "results.json"
        manifest.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            after=after,
            inputs_stable=stable,
            application_errors=errors,
            results_sha256=hashlib.sha256(
                results_path.read_bytes()
            ).hexdigest(),
            status=(
                "inputs_changed"
                if not stable
                else "application_errors"
                if errors
                else "completed"
            ),
        )
        write_manifest(manifest_path, manifest)

    except Exception as error:
        # Preserve evidence that execution did not finish successfully.
        manifest.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            failure_type=type(error).__name__,
        )
        write_manifest(manifest_path, manifest)
        raise

    print("\n".join(summary_lines(summary, settings.top_k)))
    print("\nInputs stable:", stable)
    print("Application errors:", errors)
    print("Manifest status:", manifest["status"])
    print("Report:", report_path)
    print("Provenance:", manifest_path)

    # Changed inputs must not silently become a trusted comparison baseline.
    return 0 if stable and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())