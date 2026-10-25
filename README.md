# TrueCost

### Predict AI-agent token usage before execution. Compare it with what actually happened.

TrueCost is a Penn State DS 440 team capstone studying how task characteristics and model choice affect the tokens consumed by an AI agent. We combine controlled local-LLM experiments on **ICDS Roar**, supervised regression, calibrated upper budgets, and post-execution feedback.

An agent can make several model requests and tool calls before finishing a task. TrueCost estimates **cumulative input and output tokens across that execution**, using information available before the task starts. Its point estimate supports planning; its upper budget addresses the separate question of how much capacity to reserve.

**Stack:** Python · scikit-learn · pandas · NumPy · SciPy · joblib · SQLite · Linux/Bash · Slurm · Ollama · Qwen2.5-Coder 7B/14B

[Final evaluation](ml_work/FINAL_REPORT.md) · [Frozen V1 release](releases/v1.0.0/) · [V2 research](ml_v2/README.md) · [Feedback module](truecost_feedback/README.md) · [Collection methods](docs/TrueCost_Final_Methods_Report.docx)

## Current status

| Component | Status |
|---|---|
| HPC collection and audit workflow | Completed for the V1 study; organized around a 12,000-run campaign |
| V1 point predictor and calibrated budget | Frozen; final confirmation evaluation complete |
| V2 feature and execution-risk experiments | Exploratory; no replacement for V1 promoted |
| Completion feedback | SQLite ledger, validation, reports, export, and predictor adapter implemented |
| Runner integration and web dashboard | Planned |

## Results

The frozen V1 model was evaluated once on **1,800 complete runs from 450 unseen task groups**. Models, features, and calibration parameters were fixed before confirmation evaluation.

| Confirmation metric | Result |
|---|---:|
| Mean absolute error (MAE) | **323.78 tokens** |
| Root mean squared error (RMSE) | 1,404.38 tokens |
| R² | 0.7052 |
| Mean signed error, prediction minus actual | −142.70 tokens |
| 90th-percentile absolute error | 767.83 tokens |
| Upper-budget coverage per run | **95.33%** — 1,716 / 1,800 |
| Joint upper-budget coverage across all four runs per task group | **91.56%** — 412 / 450 |
| Mean upper budget | 2,592.85 tokens |

Joint coverage means that every model/seed run for a task stayed within its respective budget. The observed 91.56% exceeded the prespecified 90% target in this sample. It is workload-specific empirical evidence; it does not establish universal or family-conditional coverage.

Large errors remain important. For the 176 confirmation runs consuming at least 5,241 tokens, MAE was 1,414.09 tokens. Coding, ideation, and tabular reasoning had joint coverage below 90%. See the [full report](ml_work/FINAL_REPORT.md) and [machine-readable metrics](ml_work/confirmation_evaluation/confirmation_metrics.csv) for family and model breakdowns.

## How it works

1. **Collect controlled executions.** Run local Qwen2.5-Coder models through a bounded six-request controller on Roar, using Slurm jobs and sharded assignments.
2. **Audit the evidence.** Check run identities, manifests, telemetry completeness, token arithmetic, and collection provenance. Preserve unsuccessful and step-limited executions when their resource measurements are complete.
3. **Build pre-execution features.** Describe task family, source, size band, model identity, and task-text structure. Exclude realized token counts, request counts, termination, latency, quality, and task/group identifiers from predictor inputs.
4. **Predict usage and budget.** Fit a histogram gradient boosting point model with absolute-error loss, plus a separate quantile model. Calibrate the upper budget using task-group residual scores.
5. **Record outcomes.** Match a prediction with measured completion telemetry, report prediction error and budget exceedance, and export eligible feedback for later review.

The frozen budget is a one-sided upper bound, not a two-sided prediction interval. Calibration leaves the point prediction unchanged.

## Data and evaluation design

The study covers eight task families: classification, coding, extraction, grounded question answering, ideation, math, summarization, and tabular reasoning. Each task group has four planned configurations: two model checkpoints × two seeds.

| Partition or stage | Evidence |
|---|---|
| Training | 8,396 complete runs from 2,100 task groups; four missing outcomes left unimputed |
| Development | Five-fold validation split by task group; split seeds 440 and 917 used for sensitivity analysis |
| Official calibration | 450 task groups used to calibrate the frozen upper-budget procedure |
| Confirmation | 450 separate task groups; 1,800 fully measured runs |

Related runs stay together in every fold. Repeated seed configurations are correlated and do not provide extra independent tasks. The campaign label “12,000 runs” should not be read as 12,000 complete, successful outcomes.

V2 investigates the expensive-run tail through trace diagnostics, feature ablations, risk classifiers, and expert routing. No tested candidate consistently improved the full set of objectives enough to replace V1. Any future candidate informed by the original confirmation results needs fresh evaluation tasks before a new final-test claim.

## Repository guide

| Path | Purpose |
|---|---|
| [`ml_work/`](ml_work/) | Working V1 scripts, frozen models, calibration, confirmation predictions, and reports |
| [`releases/v1.0.0/`](releases/v1.0.0/) | Preserved V1 package, release metadata, and artifact hashes |
| [`ml_v2/`](ml_v2/) | Exploratory diagnostics, model comparisons, and next collection plan |
| [`ml_v2/pilot_kit/`](ml_v2/pilot_kit/) | Roar collection templates, selected tasks, tests, and provenance |
| [`truecost_feedback/`](truecost_feedback/) | SQLite prediction/outcome ledger and frozen-predictor adapter |
| [`docs/`](docs/) | Data-collection methods documentation |
| [`legacy/capstone_data_pipeline/`](legacy/capstone_data_pipeline/) | Earlier public-dataset ingestion work, retained as project history |
| [`SNAPSHOT_MANIFEST.json`](SNAPSHOT_MANIFEST.json) | Inventory of the prepared project snapshot |

Start with the final report for current V1 findings. Some earlier READMEs and readiness records describe milestones that have since been completed. The preserved release records intentionally retain their original contents.

## Get started

### Clone and inspect

Access to this private repository is required.

```bash
git clone https://github.com/Kartikeya-Rana-2004/TrueCost.git
cd TrueCost
```

The saved reports and CSV results can be inspected without fitting models or accessing Roar. This repository is a research snapshot; reproducing every experiment requires external audited source exports and raw trace archives. Several research scripts retain machine-specific input paths.

### Run the feedback demo

The core feedback module uses the Python standard library and can be exercised without the ML dependencies:

```bash
cd truecost_feedback
python3 -m unittest discover -v
python3 feedback.py --db /tmp/truecost-demo.sqlite predict prediction.example.json
```

Copy the returned prediction ID into the completion command:

```bash
python3 feedback.py --db /tmp/truecost-demo.sqlite complete completion.example.json --prediction-id YOUR_ID
python3 feedback.py --db /tmp/truecost-demo.sqlite report
```

The example records contain synthetic values. Use a fresh database for a separate demo; the ledger rejects duplicate records. The module reports MAE, RMSE, signed bias, coverage, pending predictions, and incomplete telemetry. See the [feedback guide](truecost_feedback/README.md) for export rules and runner integration.

### Use the frozen predictor

The recorded ML environment uses **Python 3.12 and scikit-learn 1.8.0**, with NumPy, pandas, SciPy, and joblib. The following installs the stated dependencies; it is not a complete environment lockfile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy joblib 'scikit-learn==1.8.0'
python ml_work/predict_tasks.py --tasks /path/to/new_task_inputs.json --out /tmp/truecost-predictions.csv
```

The input is a nonempty JSON list with unique `task_id` values. Each task supplies `task_id`, `group_id`, `family`, `source_id`, `size_band`, `instruction`, and `context`, with optional `public_tests`. Match the frozen export schema and supported workload; an arbitrary prompt does not establish in-distribution validity. Existing input structure can be inspected in [`confirmation_task_inputs.json`](ml_work/confirmation_input/confirmation_task_inputs.json), whose tasks are already consumed evaluation evidence.

The CLI returns predictions for both supported model checkpoints. With the matching calibration file present, it also returns `calibrated_upper_budget`; otherwise it returns only the uncalibrated quantile estimate alongside the point prediction. The output path must not already exist. Load joblib artifacts only from trusted sources.

To register predictions for genuinely new tasks before execution:

```bash
cd truecost_feedback
python predict_and_record.py --ml-dir ../ml_work --tasks /path/to/new_task_inputs.json --db feedback.sqlite --controller-version six-request-v1
```

This adapter records predictions; the task runner must separately execute the tasks and submit measured completions. It does not automatically collect telemetry, retrain, or promote models.

## Scope and limitations

- Results apply to this study's two model checkpoints, task sources, and bounded controller. Transfer to other models, hardware, or agent policies is untested.
- The primary measured resource is total tokens. Dollar pricing, energy measurement, and validated causal cost attribution are not implemented.
- Purposive sampling and deterministic partitioning limit distribution-free interpretations of calibration.
- Post-execution telemetry supports diagnostics and evaluation; it must never become a pre-execution predictor.
- Raw Roar traces and the complete original collection environment are not bundled. Local files do not establish current remote job status.
- A dashboard, automatic runner hooks, and automatic retraining remain future work.

## Next steps

- Connect completion recording to the runner's finalization and failure-handling paths.
- Build a dashboard for predicted versus actual usage and budget exceedance.
- Collect fresh task groups focused on rare output-ceiling and execution-limit behavior, with separate calibration and untouched evaluation samples.
- Evaluate better pre-execution descriptors while preserving the frozen V1 evidence.

## Project context and licensing

TrueCost is a team capstone for **Pennsylvania State University, DS 440**, started in September 2026. This repository preserves collection infrastructure, experimental results, and ongoing development for team review.

No license has been assigned. Refer to individual upstream datasets and models for their applicable terms.
