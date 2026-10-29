"""Run CLI robustness cases and preserve responses for separate review."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_eu_law.app.config import Settings


def decode_output(value: bytes | None) -> tuple[str, bool]:
    """Preserve readable output and report invalid UTF-8 explicitly."""
    if not value:
        return "", False

    try:
        return value.decode("utf-8"), False
    except UnicodeDecodeError:
        return value.decode("utf-8", errors="replace"), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/robustness/questions.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    cases = dataset["cases"]

    # Validate the dataset before making any model requests.
    if not cases:
        raise SystemExit("The robustness dataset contains no cases.")

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate robustness case IDs.")

    for case in cases:
        for field in ("id", "question", "expected_behavior"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SystemExit(f"Case requires a nonempty {field}.")

    executable = Path(sys.executable).parent / "rag-eu-law.exe"
    if not executable.is_file():
        raise SystemExit(f"Application executable not found: {executable}")

    if args.output_dir.exists():
        raise SystemExit("Choose a new output directory to preserve prior runs.")

    print("Dataset:", dataset["version"], flush=True)
    print("Cases:", len(cases), flush=True)

    if args.dry_run:
        print("Dry run: no model requests or files written.")
        return 0

    settings = Settings()

    # Only change the child process environment; never print secrets.
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    args.output_dir.mkdir(parents=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset["version"],
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "answer_model": settings.llm_label,
        "timeout_seconds": args.timeout,
        "requested_output_encoding": "utf-8",
        "scope": "User-input robustness smoke checks; semantic review required.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    failed_execution = False
    encoding_issue = False

    for position, case in enumerate(cases, start=1):
        print(f"Running {case['id']}...", flush=True)
        started = time.perf_counter()

        # Do not use a shell: the question is a literal command argument.
        try:
            completed = subprocess.run(
                [str(executable), "ask", case["question"], "--show-chunks"],
                capture_output=True,
                env=child_env,
                timeout=args.timeout,
            )
            status = (
                "completed" if completed.returncode == 0 else "execution_error"
            )
            exit_code = completed.returncode
            stdout_bytes = completed.stdout
            stderr_bytes = completed.stderr

        except subprocess.TimeoutExpired as error:
            status = "timeout"
            exit_code = None
            stdout_bytes = error.stdout
            stderr_bytes = error.stderr

        except OSError as error:
            status = "execution_error"
            exit_code = None
            stdout_bytes = b""
            stderr_bytes = str(error).encode("utf-8")

        stdout, stdout_decode_error = decode_output(stdout_bytes)
        stderr, stderr_decode_error = decode_output(stderr_bytes)

        # Also detect replacement characters already present in valid UTF-8.
        replacement_present = "\ufffd" in stdout or "\ufffd" in stderr
        case_encoding_issue = (
            stdout_decode_error or stderr_decode_error or replacement_present
        )

        failed_execution |= status != "completed"
        encoding_issue |= case_encoding_issue

        record = {
            "case": case,
            "execution": {
                "status": status,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_decode_error": stdout_decode_error,
                "stderr_decode_error": stderr_decode_error,
                "replacement_character_present": replacement_present,
            },
            "cli_wall_time_ms": round(
                (time.perf_counter() - started) * 1000, 1
            ),
            "review": {"outcome": "pending", "reason": None},
        }

        # Save each case immediately and retain raw bytes for encoding diagnosis.
        stem = args.output_dir / f"case-{position:02d}"
        stem.with_suffix(".stdout.bin").write_bytes(stdout_bytes or b"")
        stem.with_suffix(".stderr.bin").write_bytes(stderr_bytes or b"")
        stem.with_suffix(".json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"  {status}; encoding issue: {case_encoding_issue}",
            flush=True,
        )

    print("Saved to:", args.output_dir)
    print("Semantic robustness outcomes remain pending review.")

    # This exit status covers execution and encoding, not answer correctness.
    return 1 if failed_execution or encoding_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())