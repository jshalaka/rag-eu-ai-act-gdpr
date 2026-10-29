"""Measure sequential retrieval and answer-generation pipeline latency."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.domain.constants import REFUSAL_MESSAGE
from rag_eu_law.domain.exceptions import RagError
from rag_eu_law.evaluation.loader import load_questions
from rag_eu_law.evaluation.performance_metrics import summarize_latencies
from rag_eu_law.evaluation.provenance import capture_retrieval_provenance
from rag_eu_law.generation.generator import GroundedAnswerGenerator


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class CountingClient:
    """Count logical client calls; provider-internal retries are not counted."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.delegate.complete(system_prompt, user_prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path, default=Path("eval/questions.yaml")
    )
    parser.add_argument("--ids", nargs="+", default=["d01", "c03", "x05"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if len(args.ids) != len(set(args.ids)):
        parser.error("--ids must be unique")
    if args.output_dir.exists():
        raise SystemExit("Choose a new output directory.")

    by_id = {
        question.id: question
        for question in load_questions(args.questions)
    }
    missing = set(args.ids) - by_id.keys()
    if missing:
        raise SystemExit(f"Unknown question IDs: {sorted(missing)}")

    questions = [by_id[qid] for qid in args.ids]
    print("Questions:", args.ids)
    print("Measured generation attempts:", len(questions) * args.repeats)
    print("Concurrency: 1", flush=True)

    if args.dry_run:
        print("Dry run: no retrieval, model requests, or files written.")
        return 0

    settings = Settings()
    container = Container(settings)
    retriever = container.retriever
    client = CountingClient(container.llm)
    generator = GroundedAnswerGenerator(client)

    before = capture_retrieval_provenance(settings, args.questions)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "benchmark.json"

    report = {
        "benchmark_version": "answer-performance-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": settings.llm_label,
        "reasoning_effort": (
            settings.openai_reasoning_effort
            if settings.llm_provider == "openai" else None
        ),
        "question_ids": args.ids,
        "repeats": args.repeats,
        "concurrency": 1,
        "before": before,
        "samples": [],
        "acceptance_status": "not_evaluated_no_performance_targets",
        "limitations": [
            "Small exploratory workload; not an SLA or concurrent-load test.",
            "Only retrieval is explicitly warmed before measurement.",
            "End-to-end covers retrieval and generation with citation checking; "
            "it excludes CLI/UI startup and history persistence.",
            "Logical client-call count excludes provider-internal retries.",
            "Token usage and monetary cost are not available.",
        ],
    }
    save(output, report)

    try:
        # Warm retrieval only; make no paid generation warm-up requests.
        for question in questions:
            retriever.retrieve(question.question, settings.top_k)

        rng = random.Random(42)

        for repeat in range(1, args.repeats + 1):
            ordered = list(questions)
            rng.shuffle(ordered)

            for question in ordered:
                print(f"Pass {repeat}: {question.id}", flush=True)
                calls_before = client.calls
                sample = {
                    "repeat": repeat,
                    "question_id": question.id,
                    "question": question.question,
                    "status": "running",
                    "stage": "retrieval",
                }

                started = perf_counter()
                try:
                    context = retriever.retrieve(
                        question.question, settings.top_k
                    )
                    retrieved_at = perf_counter()

                    sample.update(
                        retrieval_ms=(retrieved_at - started) * 1000,
                        context=[
                            item.model_dump(mode="json") for item in context
                        ],
                        stage="generation",
                    )

                    # Start the generation timer after context serialization.
                    generation_started = perf_counter()
                    answer = generator.generate(question.question, context)
                    finished = perf_counter()

                    sample.update(
                        status="completed",
                        stage="completed",
                        generation_ms=(finished - generation_started) * 1000,
                        # Exclude harness context-serialization time.
                        end_to_end_ms=(
                            (retrieved_at - started)
                            + (finished - generation_started)
                        ) * 1000,
                        exact_refusal=answer.answer == REFUSAL_MESSAGE,
                        answer=answer.model_dump(mode="json"),
                    )

                except RagError as error:
                    sample.update(
                        status="error",
                        error=str(error),
                        failed_attempt_wall_ms=(
                            perf_counter() - started
                        ) * 1000,
                    )

                sample["logical_model_calls"] = client.calls - calls_before
                report["samples"].append(sample)
                save(output, report)

        completed = [
            sample for sample in report["samples"]
            if sample["status"] == "completed"
        ]
        errors = len(report["samples"]) - len(completed)

        after = capture_retrieval_provenance(settings, args.questions)
        stable = before["identity_sha256"] == after["identity_sha256"]

        report.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            after=after,
            inputs_stable=stable,
            execution_errors=errors,
            execution_error_rate=errors / len(report["samples"]),
            logical_model_calls=client.calls,
            completed_refusals=sum(
                sample["exact_refusal"] for sample in completed
            ),
            latency_summaries={
                metric: summarize_latencies([
                    sample[metric] for sample in completed
                ])
                for metric in (
                    "retrieval_ms", "generation_ms", "end_to_end_ms"
                )
            },
            status="completed" if stable and not errors else "needs_review",
        )
        save(output, report)

    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__)
        save(output, report)
        raise

    print(json.dumps(report["latency_summaries"], indent=2))
    print("Execution errors:", errors)
    print("Completed refusals:", report["completed_refusals"])
    print("Inputs stable:", stable)
    print("Logical model calls:", client.calls)
    print("Saved:", output)

    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())