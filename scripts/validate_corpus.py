"""Validate chunk structure, PDF readability, and source page references."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from pydantic import ValidationError

from rag_eu_law.domain.models import Chunk


def validate_corpus(chunk_path: Path, raw_dir: Path) -> dict:
    """Collect findings without changing the corpus."""
    chunk_bytes = chunk_path.read_bytes()
    issues = []
    chunks = []
    seen_ids = set()
    regulations = Counter()
    record_count = 0

    def flag(check: str, **details) -> None:
        issues.append({"check": check, **details})

    # Validate each JSONL record against the application's chunk schema.
    for line_number, line in enumerate(
        chunk_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        record_count += 1
        try:
            chunk = Chunk.model_validate_json(line)
        except ValidationError as error:
            flag(
                "schema",
                line=line_number,
                details=[
                    {"field": list(item["loc"]), "message": item["msg"]}
                    for item in error.errors()
                ],
            )
            continue

        chunks.append(chunk)
        regulations[chunk.regulation.value] += 1

        if chunk.chunk_id in seen_ids:
            flag("duplicate_chunk_id", chunk_id=chunk.chunk_id)
        seen_ids.add(chunk.chunk_id)

        if not chunk.text.strip():
            flag("empty_text", chunk_id=chunk.chunk_id)

        locators = [chunk.article, chunk.recital, chunk.annex]
        if sum(value is not None for value in locators) != 1:
            flag("invalid_locator_count", chunk_id=chunk.chunk_id)

        if chunk.page_start < 1 or chunk.page_end < chunk.page_start:
            flag("invalid_page_range", chunk_id=chunk.chunk_id)

    if record_count == 0:
        flag("empty_chunk_file")

    # Discover PDFs, including files whose extension uses uppercase letters.
    pdf_paths = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        flag("no_source_pdfs")

    sources = []
    page_counts = {}

    for path in pdf_paths:
        source = {"file": path.name}

        try:
            source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

            with pymupdf.open(path) as document:
                source["pages"] = len(document)
                page_counts[path.name] = len(document)

                # Text availability does not establish extraction fidelity.
                source["pages_with_extractable_text"] = sum(
                    bool(page.get_text().strip()) for page in document
                )

                if len(document) == 0:
                    flag("empty_pdf", source_file=path.name)
                elif source["pages_with_extractable_text"] == 0:
                    flag("no_extractable_text", source_file=path.name)

        except (OSError, RuntimeError, ValueError) as error:
            flag(
                "pdf_read_error",
                source_file=path.name,
                error=str(error),
            )

        sources.append(source)

    discovered_names = {path.name for path in pdf_paths}

    # Chunk page positions are expected to be one-based PDF positions.
    for chunk in chunks:
        source_name = chunk.source_file

        if (
            "/" in source_name
            or "\\" in source_name
            or Path(source_name).name != source_name
        ):
            flag("source_file_not_basename", chunk_id=chunk.chunk_id)
        elif source_name not in discovered_names:
            flag(
                "missing_source_pdf",
                chunk_id=chunk.chunk_id,
                source_file=source_name,
            )
        elif source_name in page_counts:
            if not (
                1 <= chunk.page_start
                <= chunk.page_end
                <= page_counts[source_name]
            ):
                flag(
                    "page_reference_out_of_bounds",
                    chunk_id=chunk.chunk_id,
                    source_file=source_name,
                )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_file": chunk_path.as_posix(),
        "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "records_checked": record_count,
        "schema_valid_records": len(chunks),
        "regulations": dict(regulations),
        "sources": sources,
        "issue_count": len(issues),
        "issue_types": dict(Counter(item["check"] for item in issues)),
        "issues": issues,
        "passed": not issues,
        "scope": (
            "Structural validation, PDF readability, text availability, "
            "and page bounds. Semantic fidelity, locator correctness, "
            "and corpus completeness are not established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        # Preserve earlier validation evidence.
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")

        report = validate_corpus(args.chunks, args.raw_dir)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    except (OSError, ValueError) as error:
        print(f"INPUT ERROR: {error}")
        return 2

    print("Records checked:", report["records_checked"])
    print("Schema-valid records:", report["schema_valid_records"])
    print("Regulations:", report["regulations"])
    print("PDFs inspected:", len(report["sources"]))
    print("Issues found:", report["issue_count"])
    print("Issue types:", report["issue_types"])
    print("Report saved:", args.output)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())