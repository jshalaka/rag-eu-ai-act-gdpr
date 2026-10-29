"""Compare saved retrieval results without making model requests."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from rag_eu_law.evaluation.models import QuestionResult
from rag_eu_law.evaluation.regression import compare_retrieval_runs


def load_results(path: Path) -> list[QuestionResult]:
    """Validate saved records before comparing their retrieval metrics."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        QuestionResult.model_validate(item)
        for item in payload["results"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        # Keep previous comparison reports intact.
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")

        baseline = load_results(args.baseline)
        candidate = load_results(args.candidate)
        report = compare_retrieval_runs(baseline, candidate)

        report["baseline_file"] = args.baseline.as_posix()
        report["candidate_file"] = args.candidate.as_posix()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
        print(f"INPUT ERROR: {error}")
        return 2

    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print("Report saved:", args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())