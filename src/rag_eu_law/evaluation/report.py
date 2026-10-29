"""Write the evaluation report (Markdown) and raw results (JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eu_law.domain.constants import EVAL_REPORT_FILENAME, EVAL_RESULTS_FILENAME
from rag_eu_law.evaluation.models import EvalSummary, QuestionResult


def write_report(
    results: list[QuestionResult], summary: EvalSummary, directory: Path, top_k: int
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.model_dump(),
        "results": [result.model_dump(mode="json") for result in results],
    }
    (directory / EVAL_RESULTS_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = directory / EVAL_REPORT_FILENAME
    report.write_text(render_markdown(results, summary, top_k), encoding="utf-8")
    return report


def render_markdown(results: list[QuestionResult], summary: EvalSummary, top_k: int) -> str:
    lines = ["# Evaluation report", "", *summary_lines(summary, top_k), "", "## Questions", ""]
    lines += [
        "| id | type | retrieval | citation | refused | question |",
        "|---|---|---|---|---|---|",
    ]
    lines += [_row(result) for result in results]
    failures = [result for result in results if _failed(result)]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [_failure(result) for result in failures]
    return "\n".join(lines) + "\n"


def summary_lines(summary: EvalSummary, top_k: int) -> list[str]:
    return [
        "| metric | value |",
        "|---|---|",
        f"| questions (answerable / out-of-scope) | {summary.answerable_questions} / "
        f"{summary.out_of_scope_questions} |",
        f"| retrieval hit rate@{top_k} | {_pct(summary.retrieval_hit_rate_at_k)} |",
        f"| complete-source retrieval rate@{top_k} | "
        f"{_pct(summary.complete_source_retrieval_rate_at_k)} |",
        f"| mean retrieval recall@{top_k} | {_pct(summary.mean_retrieval_recall)} |",
        f"| citation accuracy | {_pct(summary.citation_accuracy)} |",
        f"| refusal accuracy (out-of-scope) | {_pct(summary.refusal_accuracy)} |",
        f"| false refusal rate (answerable) | {_pct(summary.false_refusal_rate)} |",
        f"| mean latency | {_ms(summary.mean_latency_ms)} |",
        f"| mean retrieval latency | {_ms(summary.mean_retrieval_latency_ms)} |",
        *(
            f"| hit rate@{top_k}, {name} ({kind.questions} q) | {_pct(kind.retrieval_hit_rate)} |"
            for name, kind in summary.by_type.items()
        ),
    ]


def _row(result: QuestionResult) -> str:
    return (
        f"| {result.question_id} | {result.type.value} | {_mark(result.retrieval_hit)} | "
        f"{_mark(result.citation_correct)} | {_mark(result.refused)} | {result.question} |"
    )


def _failed(result: QuestionResult) -> bool:
    out_of_scope = result.type.value == "out_of_scope"
    wrong_refusal = result.refused is not out_of_scope and result.refused is not None
    return (
        bool(result.error)
        or result.retrieval_hit is False
        or bool(result.missing_sources)
        or result.citation_correct is False
        or wrong_refusal
    )


def _failure(result: QuestionResult) -> str:
    detail = result.error or f"answer: {result.answer!r}; citations: {result.citations}"
    lines = [
        f"- **{result.question_id}** {result.question}",
        f"  - retrieved: {result.retrieved}",
    ]

    if result.missing_sources:
        labels = []
        for source in result.missing_sources:
            if source.article is not None:
                locator = f"Article {source.article}"
            elif source.recital is not None:
                locator = f"Recital {source.recital}"
            else:
                locator = f"Annex {source.annex}"
            labels.append(f"{source.regulation.value} {locator}")

        lines.append(f"  - missing expected sources: {', '.join(labels)}")

    lines.append(f"  - {detail}")
    return "\n".join(lines)


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"
