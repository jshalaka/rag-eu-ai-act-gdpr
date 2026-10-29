"""Benchmark sequential warmed retrieval without answer-model requests."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.exceptions import RagError
from rag_eu_law.evaluation.loader import load_questions
from rag_eu_law.evaluation.performance_metrics import summarize_latencies
from rag_eu_law.evaluation.provenance import capture_retrieval_provenance
from rag_eu_law.evaluation.runner import score_question


def save(path: Path, payload: dict) -> None:
    """Checkpoint the benchmark using an atomic JSON replacement."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path, default=Path("eval/questions.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup_passes < 1:
        parser.error("Repeats and warm-up passes must be positive.")
    if args.output_dir.exists():
        raise SystemExit("Choose a new output directory.")

    questions = load_questions(args.questions)
    print("Questions:", len(questions))
    print("Warm-up attempts:", len(questions) * args.warmup_passes)
    print("Measured attempts:", len(questions) * args.repeats)
    print("Concurrency: 1", flush=True)

    if args.dry_run:
        print("Dry run: no retrieval or files written.")
        return 0

    settings = Settings()

    # Construction time excludes Python process startup.
    # Any deferred initialization during retrieval falls into warm-up.
    started = perf_counter()
    retriever = Container(settings).retriever
    construction_ms = (perf_counter() - started) * 1000

    before = capture_retrieval_provenance(settings, args.questions)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "benchmark.json"

    report = {
        "benchmark_version": "retrieval-performance-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "concurrency": 1,
        "seed": 42,
        "question_count": len(questions),
        "warmup_passes": args.warmup_passes,
        "measured_passes": args.repeats,
        "retriever_construction_ms": construction_ms,
        "warmup_total_ms": None,
        "before": before,
        "after": None,
        "inputs_stable": None,
        "samples": [],
        "acceptance_status": "not_evaluated_no_performance_targets",
    }
    save(output, report)

    try:
        # Warm-up requests are excluded from measured latency distributions.
        started = perf_counter()
        for _ in range(args.warmup_passes):
            for question in questions:
                retriever.retrieve(question.question, settings.top_k)
        report["warmup_total_ms"] = (perf_counter() - started) * 1000

        rng = random.Random(42)

        for repeat in range(1, args.repeats + 1):
            ordered = list(questions)
            rng.shuffle(ordered)

            for question in ordered:
                sample = {
                    "repeat": repeat,
                    "question_id": question.id,
                    "type": question.type.value,
                    "status": None,
                    "error": None,
                }

                started = perf_counter()
                try:
                    context = retriever.retrieve(
                        question.question, settings.top_k
                    )
                except RagError as error:
                    sample.update(
                        status="error",
                        error=str(error),
                        latency_ms=(perf_counter() - started) * 1000,
                    )
                else:
                    elapsed_ms = (perf_counter() - started) * 1000

                    # Score outside the timed retrieval operation.
                    scored = score_question(question, context, None)
                    sample.update(
                        status="success",
                        latency_ms=elapsed_ms,
                        retrieved=scored.retrieved,
                        retrieval_hit=scored.retrieval_hit,
                        retrieval_recall=scored.retrieval_recall,
                        empty_context=not context,
                    )

                report["samples"].append(sample)

            save(output, report)
            print(f"Measured pass {repeat} saved.", flush=True)

        after = capture_retrieval_provenance(settings, args.questions)
        stable = before["identity_sha256"] == after["identity_sha256"]

        successes = [
            sample for sample in report["samples"]
            if sample["status"] == "success"
        ]
        failures = [
            sample for sample in report["samples"]
            if sample["status"] == "error"
        ]

        # Ordered chunk-ID agreement is an observation of repeatability,
        # rather than proof of deterministic retrieval.
        changed_questions = []
        for question in questions:
            observations = [
                sample for sample in successes
                if sample["question_id"] == question.id
            ]
            sequences = {
                tuple(sample["retrieved"]) for sample in observations
            }
            if len(sequences) > 1:
                changed_questions.append(question.id)

        report.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            after=after,
            inputs_stable=stable,
            execution_errors=len(failures),
            execution_error_rate=len(failures) / len(report["samples"]),
            empty_successful_contexts=sum(
                sample["empty_context"] for sample in successes
            ),
            successful_retrieval_latency=summarize_latencies(
                [sample["latency_ms"] for sample in successes]
            ),
            failed_attempt_latency=summarize_latencies(
                [sample["latency_ms"] for sample in failures]
            ),
            questions_with_changed_ordered_chunk_ids=changed_questions,
            status="completed" if stable and not failures else "needs_review",
        )
        save(output, report)

    except Exception as error:
        # Keep evidence of interrupted or failed execution.
        report.update(
            status="failed",
            failure_type=type(error).__name__,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        save(output, report)
        raise

    print("\nSuccessful retrieval latency:")
    print(json.dumps(report["successful_retrieval_latency"], indent=2))
    print("Execution errors:", report["execution_errors"])
    print("Inputs stable:", report["inputs_stable"])
    print("Changed chunk-ID order:", changed_questions)
    print("Performance acceptance:", report["acceptance_status"])
    print("Saved:", output)

    # Exit status covers execution/input validity, not an unapproved SLA.
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())