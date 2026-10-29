# RAG evaluation progress tracker

Last consolidated: 28 October 2026. Owner: Shalaka.

## Current milestone

The initial evaluation framework is implemented on `feature/rag-evaluation-framework`.
The Windows CI-style run reported **426 passed, 10 skipped, 97% statement coverage, no warnings,
exit code 0**. These are user-reported local results, not an independently rerun GitHub result.
At this audit, GitHub showed the feature branch but no pull request or workflow executions.

Status meanings: **baseline complete** covers only the initial measured scope;
**in progress** means agreed broader exit criteria remain. No overall completion percentage
is assigned. Passing software tests does not establish RAG answer quality or release readiness.

## Phase dashboard

| Phase | Status | Verified initial scope | Remaining work |
|---|---|---|---|
| 1. Baseline/provenance | Baseline complete | Version-2 manifests; logical Chroma fingerprint; incompatible/tampered-input rejection tests | Pin embedding artifacts; ingestion lineage; script/prompt identity beyond captured fields; independent reproducibility |
| 2. Dataset | In progress | 26 questions, expected sources; approved three-fact d01 pilot | Broader reviewed references and independent source-label review |
| 3. Ingestion/chunks | Baseline complete | 1,154 structural checks; two PDFs; three recorded fidelity samples; reusable validator | Risk-based fidelity/boundary sampling and completeness checks |
| 4. Retrieval | Baseline complete | Hit/recall/complete-source metrics; aggregate and per-question gates; compatible run comparison | Rank-sensitive metrics, broader dataset and release acceptance policy |
| 5. Answer/citation evaluation | In progress | Grounding judge, checkpoint runner, review-aware reports, three-fact coverage pilot | Independent calibration, wider correctness/completeness and reusable reference commands |
| 6. Robustness/security | Baseline complete | Four user-input cases; four controlled generation cases; five deterministic control tests | More source attacks, repeated trials and end-to-end attack paths |
| 7. Performance/reliability | In progress | 78 measured retrievals and nine answer attempts; stable inputs and zero observed errors | Approved targets, load, cold starts, timeouts/retries and cost |
| 8. Delivery/CI | In progress | Runbook, tests, synthetic negative gate, workflow prepared, local CI-style pass | Open PR; verify Ubuntu/Windows clean installs and jobs; locked dependencies; lint/type audit |

## Recorded findings

| Measure | Observation | Scope |
|---|---|---|
| Retrieval hit rate@5 | 100% | 20 answerable questions |
| Complete-source rate@5 | 65% | 13/20 answerable questions |
| Mean expected-source recall@5 | Approximately 84% | Rounded display; compare JSON precision |
| Expected-source citation match | 80% | Captured full baseline; not semantic support |
| False refusal / correct out-of-scope refusal | 10% / 100% | Recorded full run; small dataset |
| Grounding verdicts | 96 supported, nine insufficient evidence | 105 model-extracted claims; provisional |
| Strict reference-fact coverage | Two covered, one partial | Three d01 overview facts only |
| Warm retrieval timing | Median 55.9 ms; p95 91.8 ms | 78 attempts, concurrency one, zero errors |
| Answer pipeline timing | Median 3.21 s; maximum 5.34 s | Nine attempts, three questions, zero errors/refusals |

Earlier model-comparison reports under `eval/local-qwen2.5-7b/` and `eval/luna/` describe different
runs. Current raw run/calibration evidence remains local, not committed in this feature branch.
See [runbook](RUNBOOK.md) and [evidence policy](../../eval/README.md).

## Immediate backlog

- [x] Implement claim-level human-review integration; 19 focused report tests passed.
- [ ] Recheck rendered claim sections against the latest local human-review JSON; unresolved summary-qualification findings must remain visible.
- [x] Add resumable judging with input, prompt/schema and configuration identity; checkpoint preservation and changed-input rejection observed.
- [x] Package corpus validation and UTF-8 robustness execution; smoke runs completed without reported encoding issues.
- [x] Exercise a degraded retrieval candidate and verify CLI exit code 1.
- [x] Add and validate three approved reference facts for a scoped d01 pilot.
- [ ] Package reference judging/reporting into reusable commands.
- [x] Sample five supported claims in an assistant-assisted pilot.
- [ ] Expand independent calibration; do not present copied or assistant-assisted reviews as independent evidence.
- [x] Implement version-2 provenance compatibility and per-question retrieval comparisons; 8 manifested-gate tests passed.
- [x] Add initial evidence-absent and source-injection checks; broader/repeated security work remains.
- [x] Measure initial retrieval and answer latency distributions.
- [ ] Approve performance/reliability targets and measure load, cold starts, retries/timeouts and cost.
- [x] Resolve the observed SQLite connection warnings; classify ten skipped model-download tests.
- [ ] Execute the gated real-model contract tests when deliberately scheduled.
- [x] Prepare GitHub workflow and verify local CI-style configuration.
- [ ] Open the pull request and verify Ubuntu/Windows jobs and uploaded evidence.
- [ ] Revalidate lint/strict types and select a locked dependency-installation strategy.
- [ ] Select, review and publish portable baseline evidence if needed; raw local runs are currently excluded.

## Limitations

The AI Act document is a first-reading text, not the final Official Journal regulation.
Citation existence differs from claim support; grounding differs from completeness and legal
correctness. The generator and initial judge used the same configured model. Refusal wording
must be assessed separately from dataset-based refusal correctness. No independent judge
accuracy estimate, comprehensive security assessment, approved SLA or production assurance
has been established. Review counts must be read from the latest local JSON rather than
inferred from a prior 'done' message.

## Milestone record

| Date | Milestone | Evidence |
|---|---|---|
| 2026-10-25 | Initial baseline | 26 answers/judge assessments; structural/source checks; three fidelity samples |
| 2026-10-28 | SQLite cleanup | Subsequent full tests reported no resource warnings |
| 2026-10-28 | Retrieval/provenance | Version-2 comparison passed seven checks; physical SQLite differences diagnostic |
| 2026-10-28 | Generation security pilot | Four controlled cases reviewed against displayed raw/final outputs |
| 2026-10-28 | Performance pilot | 78 retrieval and nine answer measurements, sequential |
| 2026-10-28 | Local CI-style run | 426 passed, ten skipped; 97% coverage; exit code 0 |
| 2026-10-28 | Feature publication | Commit 82fedab on feature/rag-evaluation-framework; no PR/CI run visible at audit |

Update the affected phase and task after each step. Record exact commands, output paths,
review provenance and remaining exit criteria. Preserve earlier run files; use fresh paths.
