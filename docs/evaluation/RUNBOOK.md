# RAG evaluation runbook

Updated 28 October 2026. Run commands from the repository root in PowerShell.
The purpose is to evaluate the application, preserving failures rather than changing the
application or expected sources to improve a score.

## Prerequisites

Install the application and test dependencies as described in the [README](../../README.md).
The full CI dependency set is `.[index,index-lite,llm,api,ui,cloud]` plus `pytest pytest-cov`.
Keep provider keys in the local `.env`. Ingest the PDFs before commands that need chunks or
indexes. Stop other index writers during provenance capture and retrieval comparisons.
Use a fresh output directory or filename; several commands deliberately reject overwrites.

## Software tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs --tb=short -W default -o filterwarnings=
```

Latest local CI-style verification: 426 passed, ten model-download-gated tests skipped,
97% statement coverage, no warnings reported, exit code 0. Coverage is not answer accuracy.
Set `RAG_RUN_SLOW=1` only when deliberately running the real-model contract tests; these
were not executed in the latest local run. Lint and strict typing have not been revalidated
for the full evaluation addition.

## Corpus validation

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_corpus.py `
    --output .\eval\runs\corpus-review-001.json
```

Checks schema, IDs, text presence, source PDF readability and page references. The recorded
baseline had 1,154 chunks (700 AI Act, 454 GDPR), two PDFs, and zero issues in implemented
checks. Three user-recorded fidelity samples passed; this does not prove corpus-wide fidelity.

## Retrieval and provenance

```powershell
.\.venv\Scripts\python.exe .\scripts\run_retrieval_evaluation.py `
    --output-dir .\eval\runs\retrieval-review-001

.\.venv\Scripts\python.exe .\scripts\run_retrieval_evaluation.py `
    --output-dir .\eval\runs\retrieval-review-002

.\.venv\Scripts\python.exe .\scripts\check_manifested_regression.py `
    --baseline .\eval\runs\retrieval-review-001\results.json `
    --candidate .\eval\runs\retrieval-review-002\results.json `
    --output .\eval\runs\retrieval-review-comparison-001.json

$LASTEXITCODE
```

Each run saves `results.json`, `report.md` and `provenance.json`. The manifested gate requires
adjacent completed, stable version-2 provenance and the exact results-file digest. It checks
question expectations, corpus, logical vector content/configuration, non-SQLite index files,
retrieval settings, application source files and environment. Physical SQLite differences are
diagnostic; logical content changes remain incompatible. See [architecture](ARCHITECTURE.md).

Exit codes for the manifested gate: 0 passes, 1 retrieval regression/failed score check,
2 invalid or incompatible inputs. Aggregate and per-question checks both apply. A synthetic
degraded candidate was separately observed to fail with exit code 1.

The older `check_retrieval_regression.py` accepts runs without manifested provenance; retain
it for legacy comparisons. Do not describe that command as provenance enforcement.

## Full application evaluation and grounding judge

These commands use the configured answer/judge model and may incur API charges.

```powershell
.\.venv\Scripts\rag-eu-law.exe eval --output-dir .\eval\runs\answers-review-001

.\.venv\Scripts\python.exe .\scripts\judge_saved_results.py `
    --input .\eval\runs\answers-review-001\results.json `
    --output-dir .\eval\runs\judge-review-001 --dry-run

.\.venv\Scripts\python.exe .\scripts\judge_saved_results.py `
    --input .\eval\runs\answers-review-001\results.json `
    --output-dir .\eval\runs\judge-review-001
```

The judge runner saves per-record checkpoints and can resume using the same command.
`--limit 1` processes at most one pending record. Changed inputs or judge configuration require
a new output directory. Check the saved batch-complete flag; exit code 0 alone does not mean
all records were processed when a limit was used. There is no automatic human-review authoring
CLI; retain the review JSON and its provenance separately.

## Combined report

```powershell
.\.venv\Scripts\python.exe .\scripts\build_evaluation_report.py `
    --run-dir .\eval\runs\judge-review-001 `
    --output .\eval\runs\judge-review-001\combined-report-001.md
```

The report builder expects `results.json` and `judge-batch-001.json` in the run directory.
It optionally reads `human-review-001.json`, verifies input hashes/rubric identity and displays
reviews against original claim positions. Pending, missing and unresolved reviews remain
visible; judge verdicts are preserved. No model requests are made by the report builder.

## Robustness and controlled generation security

```powershell
.\.venv\Scripts\python.exe .\scripts\run_robustness.py `
    --output-dir .\eval\runs\robustness-review-001 --dry-run
.\.venv\Scripts\python.exe .\scripts\run_generation_security.py `
    --output-dir .\eval\runs\security-review-001 --dry-run
```

Remove `--dry-run` to execute. Robustness runs four full CLI cases with UTF-8 capture and
per-case persistence. Controlled generation security runs four cases, with three expected
model calls, bypassing ingestion/retrieval. Inspect raw and final responses, context and
citations. A missing marker or exit code 0 is not a semantic security pass. Save review
findings separately and label assistant-assisted review as such.

## Performance

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_retrieval.py `
    --output-dir .\eval\runs\retrieval-benchmark-review-001 --dry-run
.\.venv\Scripts\python.exe .\scripts\benchmark_answers.py `
    --output-dir .\eval\runs\answer-benchmark-review-001 --dry-run
```

Remove `--dry-run` to measure. Default retrieval benchmark: 26 warm-ups and 78 measured
attempts, sequential. Default answer benchmark: nine attempts across d01/c03/x05, sequential,
with retrieval warm-up but no generation warm-up. Pipeline timing combines retrieval and
generation durations and excludes CLI/UI/history and harness serialization. Nearest-rank p95
on nine samples is the maximum. Targets, load, provider retries and cost remain unmeasured.

## Reference coverage and calibration

`eval/references/pilot-v1.json` contains three approved overview facts for d01. The loader checks
corpus digest, question identity, evidence chunks and approval metadata. The reference judge
and metrics are implemented, but reusable reference-assessment/report commands remain a gap;
the first assessment was produced through a terminal snippet. Do not report its 2/3 fully
covered facts as overall answer accuracy or exhaustive Article 17 completeness.

Supported-claim sampling and flagged-claim reviews were assistant-assisted pilots. Review files
and raw run evidence remain local. A fresh checkout cannot reproduce their reports without
those artifacts or a new application/judge execution. See [evidence policy](../../eval/README.md).

## GitHub CI

The workflow runs software tests on Ubuntu 24.04 and Windows 2022, using Python 3.13. It installs
optional backends, checks dependency consistency and Python syntax, then uploads JUnit,
coverage and dependency-version evidence. Model-download tests are disabled; no paid model
benchmark is invoked. Dependency installation uses open ranges rather than the committed uv
lockfile, so locked CI reproducibility is still outstanding.

The workflow triggers on pull requests and pushes to main/master. A feature-branch push alone
does not trigger this push filter. At the repository audit on 28 October 2026, no pull request
or workflow runs were visible; verify actual checks after opening the pull request.
