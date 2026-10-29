"""Command line interface: `ingest`, `ask`, `eval`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from rag_eu_law.app.config import Settings
from rag_eu_law.app.container import Container
from rag_eu_law.app.logging_config import configure_logging
from rag_eu_law.domain import constants
from rag_eu_law.domain.exceptions import RagError
from rag_eu_law.domain.models import Answer, Regulation, ScoredChunk, SearchFilter
from rag_eu_law.evaluation.loader import load_questions
from rag_eu_law.evaluation.metrics import summarize
from rag_eu_law.evaluation.report import summary_lines, write_report
from rag_eu_law.evaluation.runner import EvalRunner
from rag_eu_law.ingestion.pipeline import write_chunks

ContainerFactory = Callable[[], Container]
EXIT_FAILURE = 1


class RegulationChoice(StrEnum):
    GDPR = "gdpr"
    AI_ACT = "ai-act"

    def to_regulation(self) -> Regulation:
        return Regulation.GDPR if self is RegulationChoice.GDPR else Regulation.AI_ACT


def default_container() -> Container:
    settings = Settings()
    configure_logging(settings.log_level)
    return Container(settings)


def create_app(container_factory: ContainerFactory = default_container) -> typer.Typer:
    """Build the CLI. The container factory is injectable so tests can use fakes."""
    app = typer.Typer(help="Cited Q&A over the GDPR and the EU AI Act.", no_args_is_help=True)

    @app.command()
    def ingest(
        raw_dir: Annotated[Path | None, typer.Option(help="Folder with the PDFs.")] = None,
    ) -> None:
        """Read the PDFs, split them by legal structure and build the search indexes."""
        with _failures_as_exit():
            container = container_factory()
            chunks = container.ingestion_pipeline.ingest_directory(
                raw_dir or container.settings.raw_dir
            )
            write_chunks(chunks, container.settings.processed_dir)
            container.indexing_pipeline.index(chunks)
        counts = Counter(chunk.regulation.value for chunk in chunks)
        summary = ", ".join(f"{name}: {number}" for name, number in sorted(counts.items()))
        typer.echo(f"Ingested {len(chunks)} chunks ({summary}).")

    @app.command()
    def ask(
        question: Annotated[str, typer.Argument(help="Your question.")],
        regulation: Annotated[RegulationChoice | None, typer.Option(help="Only this law.")] = None,
        article: Annotated[str | None, typer.Option(help="Only this article number.")] = None,
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the full answer as JSON.")
        ] = False,
        show_chunks: Annotated[bool, typer.Option(help="List the retrieved chunks.")] = False,
    ) -> None:
        """Answer a question from the ingested texts, with citations."""
        search_filter = SearchFilter(
            regulation=regulation.to_regulation() if regulation else None, article=article
        )
        with _failures_as_exit():
            answer = container_factory().ask_service.ask(question, search_filter)
        typer.echo(
            answer.model_dump_json(indent=2) if as_json else _format_answer(answer, show_chunks)
        )

    @app.command("eval")
    def evaluate(
        questions: Annotated[Path, typer.Option(help="YAML question set.")] = Path(
            constants.DEFAULT_QUESTIONS_PATH
        ),
        output_dir: Annotated[
            Path, typer.Option(help="Where report.md and results.json go.")
        ] = Path(constants.DEFAULT_EVAL_OUTPUT_DIR),
        retrieval_only: Annotated[
            bool, typer.Option(help="Skip the LLM; measure retrieval only.")
        ] = False,
    ) -> None:
        """Measure retrieval hit rate, citation accuracy and refusal accuracy."""
        with _failures_as_exit():
            container = container_factory()
            generator = None if retrieval_only else container.answer_generator
            top_k = container.settings.top_k
            results = EvalRunner(container.retriever, generator, top_k).run(
                load_questions(questions)
            )
            summary = summarize(results)
            report = write_report(results, summary, output_dir, top_k)
        typer.echo("\n".join(summary_lines(summary, top_k)))
        typer.echo(f"\nReport written to {report}")

    return app


def _format_answer(answer: Answer, show_chunks: bool) -> str:
    lines = [answer.answer, ""]
    if answer.citations:
        lines += ["Citations:", *(f"  {citation.format()}" for citation in answer.citations), ""]
    if show_chunks:
        lines += ["Retrieved:", *(_chunk_line(item) for item in answer.retrieved_chunks), ""]
    lines += [f"({answer.latency_ms:.0f} ms)", answer.disclaimer]
    return "\n".join(lines)


def _chunk_line(item: ScoredChunk) -> str:
    return f"  {item.score:.3f}  {item.chunk.heading}  (p. {item.chunk.page_start})"


@contextmanager
def _failures_as_exit() -> Iterator[None]:
    """Turn expected project errors into a one-line message and exit code 1."""
    try:
        yield
    except RagError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from error


app = create_app()
