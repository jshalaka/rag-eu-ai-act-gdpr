"""Judge saved answers with checkpoints and safe resumption."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    LLMJudge,
    RUBRIC_VERSION,
)
from rag_eu_law.evaluation.judge_models import JudgeAssessment
from rag_eu_law.evaluation.judge_runner import judge_results
from rag_eu_law.evaluation.models import QuestionResult


def write_json(path: Path, payload: dict) -> None:
    """Replace a generated file atomically to avoid partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending work without making model requests or writing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum pending records to process in this invocation.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    source_bytes = args.input.read_bytes()
    payload = json.loads(source_bytes)
    results = [
        QuestionResult.model_validate(item)
        for item in payload["results"]
    ]

    ids = [result.question_id for result in results]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question IDs in application results.")

    settings = Settings()
    schema_text = json.dumps(
        JudgeAssessment.model_json_schema(),
        sort_keys=True,
    )

    # Refuse to reuse checkpoints if the input or judge configuration changes.
    identity = {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "judge_model": settings.llm_label,
        "rubric_version": RUBRIC_VERSION,
        "prompt_sha256": hashlib.sha256(
            JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            schema_text.encode("utf-8")
        ).hexdigest(),
        "reasoning_effort": (
            settings.openai_reasoning_effort
            if settings.llm_provider == "openai"
            else None
        ),
    }

    folder = args.output_dir
    manifest_path = folder / "manifest.json"
    checkpoints = folder / "checkpoints"

    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest["identity"] != identity:
            raise SystemExit(
                "Inputs or judge configuration changed. "
                "Choose a new output directory."
            )
    else:
        # Do not adopt an unrelated directory or overwrite previous evidence.
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(
                "Output directory is nonempty but has no compatible manifest."
            )
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
        }

    completed = {}
    pending = []

    for position, result in enumerate(results, start=1):
        path = checkpoints / f"result-{position:04d}.json"

        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["question_id"] != result.question_id:
                raise SystemExit(f"Checkpoint question mismatch: {path}")
            completed[result.question_id] = record
        else:
            pending.append((position, result))

    print("Total records:", len(results), flush=True)
    print("Saved checkpoints:", len(completed), flush=True)
    print("Pending records:", len(pending), flush=True)

    if args.dry_run:
        print("Dry run: no model requests or files written.")
        return

    folder.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(exist_ok=True)

    if not manifest_path.exists():
        write_json(manifest_path, manifest)

    # Keep an exact input snapshot so the existing report command can read it.
    snapshot = folder / "results.json"
    if snapshot.exists():
        if snapshot.read_bytes() != source_bytes:
            raise SystemExit("Output input snapshot differs from the source.")
    else:
        snapshot.write_bytes(source_bytes)

    selected = pending[:args.limit] if args.limit else pending

    # Construct the client only when there is work to perform.
    if selected:
        judge = LLMJudge(Container(settings).llm)

        for position, result in selected:
            print(f"Judging {result.question_id}...", flush=True)

            record = judge_results(
                [result],
                judge,
                judge_model=settings.llm_label,
            )[0]

            write_json(
                checkpoints / f"result-{position:04d}.json",
                record,
            )
            completed[result.question_id] = record
            print(f"  {record['status']}; checkpoint saved", flush=True)

    # Preserve input ordering when rebuilding the batch report.
    records = [
        completed[result.question_id]
        for result in results
        if result.question_id in completed
    ]

    report = {
        **identity,
        "created_at": manifest["created_at"],
        "source_file": args.input.as_posix(),
        "complete": len(records) == len(results),
        "results": records,
    }
    write_json(folder / "judge-batch-001.json", report)

    print("Statuses:", dict(Counter(r["status"] for r in records)))
    print("Batch complete:", report["complete"])


if __name__ == "__main__":
    main()