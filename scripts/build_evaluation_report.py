"""Build a report from saved application, judge, and human-review results."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rag_eu_law.evaluation.judge_report import render_judge_report
from rag_eu_law.evaluation.models import QuestionResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("eval/runs/judge-input-001"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination Markdown file; existing files are preserved.",
    )
    args = parser.parse_args()

    folder = args.run_dir
    output = args.output or folder / "baseline-report-reviewed-001.md"

    # Preserve reports generated during earlier milestones.
    if output.exists():
        raise SystemExit(f"Report already exists: {output}")

    source_bytes = (folder / "results.json").read_bytes()
    application = json.loads(source_bytes)
    judged = json.loads(
        (folder / "judge-batch-001.json").read_text(encoding="utf-8")
    )

    # Assessments must belong to this exact application-results file.
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != judged["source_sha256"]:
        raise SystemExit("Application results do not match the judge input hash.")

    results = [
        QuestionResult.model_validate(item)
        for item in application["results"]
    ]

    # Load reviews before rendering so individual claim sections can use them.
    reviews = []
    review_file = folder / "human-review-001.json"

    if review_file.exists():
        review_payload = json.loads(
            review_file.read_text(encoding="utf-8")
        )

        if review_payload["source_sha256"] != source_hash:
            raise SystemExit(
                "Human review belongs to different application data."
            )

        if review_payload["rubric_version"] != judged["rubric_version"]:
            raise SystemExit("Human review and judge rubric versions differ.")

        reviews = review_payload["reviews"]

    # The renderer validates review IDs and completed review fields.
    # Human decisions are displayed separately; judge verdicts stay unchanged.
    report = render_judge_report(
        results,
        judged["results"],
        reviews=reviews,
    )

    report += (
        "\n## Evaluation provenance\n\n"
        f"- Answer-results file: {folder / 'results.json'}\n"
        f"- Judge model: {judged['judge_model']}\n"
        f"- Rubric version: {judged['rubric_version']}\n"
        f"- Judge run timestamp: {judged['created_at']}\n"
        f"- Source SHA-256: {source_hash}\n"
        f"- Human-review file present: {review_file.exists()}\n"
    )

    statuses = Counter(
        item["human_review"]["status"] for item in reviews
    )
    decisions = Counter(
        item["human_review"]["decision"]
        for item in reviews
        if item["human_review"]["status"] == "reviewed"
    )

    report += (
        "\n## Human-review progress\n\n"
        f"- Marked reviewed: {statuses['reviewed']}\n"
        f"- Pending records: {statuses['pending']}\n"
        f"- Agree with judge: {decisions['agree_with_judge']}\n"
        f"- Disagree with judge: {decisions['disagree_with_judge']}\n"
        f"- Need further adjudication: {decisions['needs_more_evidence']}\n\n"
        "Missing review records appear as pending in individual claim sections. "
        "Reviewed cases may still be unresolved. "
        "These counts do not establish judge accuracy or independent validation. "
        "The review sample contains flagged claims only.\n"
    )

    # Write only after all validation and rendering have succeeded.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Report saved: {output}")
    print(f"Reviewed records: {statuses['reviewed']}")
    print(f"Pending records: {statuses['pending']}")
    print(
        "Reviewed but unresolved:",
        decisions["needs_more_evidence"],
    )


if __name__ == "__main__":
    main()