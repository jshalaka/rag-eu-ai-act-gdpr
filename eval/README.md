# Evaluation datasets and evidence

| Path | Purpose | Repository policy |
|---|---|---|
| `questions.yaml` | 26 questions and expected source labels | Versioned input |
| `references/pilot-v1.json` | Three approved overview facts for d01 and reference chunks | Versioned pilot input; not exhaustive ground truth |
| `robustness/questions.json` | Four user-input robustness cases | Versioned input |
| `local-qwen2.5-7b/`, `luna/` | Earlier model-comparison results | Historical evidence; preserve run-specific interpretation |
| `report.md`, `results.json` | Earlier default outputs | Legacy snapshots, not automatically the latest baseline |
| `runs/` | New application/judge/security/performance results and checkpoints | Local generated evidence; ignored by default |
| `calibration/` | Sampled claims and assistant-assisted review records | Local evidence; ignored by default |

The initial evaluation work captured additional runs locally. Those files were deliberately not
included in commit 82fedab. Documentation references to these local paths are evidence locations,
not claims that the files can be downloaded from this repository.

## Reproduce or share evidence

Follow the [runbook](../docs/evaluation/RUNBOOK.md) to create new runs. A clean checkout needs
index ingestion and configured model access before live evaluations; reports requiring saved
judge/review data cannot be regenerated from software tests alone.

For a portable baseline, select a dedicated versioned fixture or release artifact after reviewing
its contents. Include the dataset/corpus identity, settings, rubric/model identity, input hashes,
limitations and review provenance. Do not bulk-add local run directories. A supported-claim
sample and assistant-authored reviews are calibration inputs, not independent validation.

## Interpret scores

- Hit rate asks whether at least one expected source was retrieved.
- Recall asks what fraction of expected sources were retrieved.
- Complete-source rate asks whether all expected sources were found without an application error.
- Citation matching checks reference identity; it is not claim entailment.
- Grounding judgments compare claims to retrieved context; they remain model judgments.
- Reference coverage checks approved required facts, currently only three for one question.
- Refusal correctness uses dataset scope separately from literal refusal wording.
- Timing observations have no approved acceptance target; nine-sample p95 is the maximum.

The AI Act corpus is a Parliament first-reading document. Scores apply to the supplied corpus,
not automatically to the final regulation or current law.
