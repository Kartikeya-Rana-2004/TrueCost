# TrueCost — Penn State DS 440 capstone

Research on pre-execution AI-agent token usage prediction and calibrated upper budgets, with controlled local-LLM experiments on Penn State ICDS Roar.

## Current progress

- **V1 frozen evaluation completed:** 8,396 complete training runs; confirmation evaluation on 1,800 runs from 450 unseen task groups. Confirmation MAE **323.78 tokens**, RMSE **1,404.38**, R² **0.7052**. Joint all-four-run task-group upper-budget coverage **91.56%**, per-run coverage **95.33%**. These are empirical workload-specific results, not universal coverage guarantees.
- **V2 development:** trace diagnostics, feature ablations, execution-risk classifiers, and expert-routing comparisons. V1 remains the usable checkpoint; exploratory V2 results do not establish an improved final predictor.
- **Roar pilot kit prepared:** separately versioned collection scripts and provenance. Local artifacts do not establish the latest remote job status.
- **Completion feedback implemented:** SQLite prediction/outcome ledger, error metrics, JSON export, and predictor adapter. Runner integration and a web dashboard remain future work.

## Repository layout

| Directory | Contents |
|---|---|
| `ml_work/` | V1 training, calibration, final evaluation, model artifacts and results |
| `releases/v1.0.0/` | Preserved frozen V1 package and hashes |
| `ml_v2/` | V2 experiments, diagnostics, pilot kit and local source snapshots |
| `truecost_feedback/` | Completion-feedback module and tests |
| `docs/` | Collection methods report |
| `legacy/capstone_data_pipeline/` | Earlier public-dataset ingestion pipeline; historical research direction |

Read `ml_work/FINAL_REPORT.md` for the final evaluation, `ml_v2/README.md` for development findings, and `truecost_feedback/README.md` for integration instructions. Historical reports may describe earlier milestones; current limitations are stated in these documents.

## Reproduction and boundaries

The ML environment uses Python, NumPy, pandas, SciPy, scikit-learn 1.8.0 and joblib. Some scripts retain original local input paths and require external audited datasets or trace archives; this snapshot is not a self-contained reproduction of every experiment. Consult individual READMEs before execution. No full raw Roar trace archive or live remote state is included unless separately supplied. Model artifacts are Python joblib files and should be loaded only from trusted sources.

The final predictor uses pre-execution descriptors, excluding realized execution telemetry. Task groups are the separation unit. V2 tuning requires fresh evaluation tasks before any new final-test claims. The work does not establish causal cost decomposition, energy usage, or dollar costs for local models.

## Feedback tests

```bash
cd truecost_feedback
python3 -m unittest discover -v
```

Snapshot prepared October 24, 2026. Source reference files were copied without modifying synced project originals. No license is assigned by this snapshot.
