# TrueCost ML development update — October 21, 2026

> Current status: official calibration is now complete. See `official_calibration/README.md` for the correction, diagnostics, locked analysis protocol, and next required confirmation export. The development report below is retained as history. Keep the selected models frozen; do not rerun model selection or `finalize_development.py` as part of confirmation evaluation.

We improved prediction of cumulative total tokens under the frozen, six-request controller. The selected development candidate is histogram gradient boosting with absolute-error loss, richer pre-execution task descriptors, and model identity. Models are saved for continued development; official calibration and final confirmation evaluation are pending.

## Measured improvement

These comparisons use identical task-group folds for each candidate and its rerun baseline. Each task's models and seeds remain together. The second fold seed is a sensitivity analysis on the same training tasks, not an independent test set.

| Five-fold split seed | Rerun forest MAE | Selected model MAE | Reduction |
|---|---:|---:|---:|
| 440 | 396.34 | 359.98 | 9.17% |
| 917 | 402.10 | 359.44 | 10.61% |

For split 440, RMSE improved from 1,780.64 to 1,763.68 tokens, R² from 0.583 to 0.591, and 90th-percentile absolute error from 905.23 to 808.96 tokens. Tail MAE improved from 1,878.47 to 1,691.71 tokens. However, tail signed error worsened from −1,376.96 to −1,570.44 tokens; average overall bias also became more negative. The predictor is better at typical absolute error and still underestimates costly runs. Point accuracy and a conservative budget are distinct objectives.

The earlier saved forest benchmark had MAE 395.76; the matched rerun here is 396.34. We use the rerun for improvement percentages. The small benchmark variation remains unexplained; we do not claim bit-for-bit reproduction of that older output.

Task-cluster resampling (2,000 resamples per split) gives descriptive 95% intervals for mean absolute-error reduction of 26.89–46.09 tokens and 30.95–54.44 tokens. These reflect variation within the collected tasks; purposive sampling and model selection prevent interpreting them as population or final-test guarantees.

## What changed and what was rejected

- Added pre-run text structure: digit/punctuation fractions, unique-word ratio, line and question counts, numeric expressions, sentence markers, table separators, coding markers, public-test length, and length ratios/transforms.
- Used absolute-error loss on raw total tokens, aligning training with the primary MAE objective.
- Tested more flexible forests, Extra Trees, log-target boosting, sparse TF-IDF ridge, fixed tree/text blends, and models specialized by family/model. The pooled absolute-loss model gave the most consistent MAE performance across the two fold sets. Text-only ridge and raw squared-loss boosting performed substantially worse.
- An ablation using only the old basic descriptors with absolute-loss boosting achieved MAE 375.54/373.76. Rich descriptors provided a further improvement to 359.98/359.44. Removing model identity increased MAE to 604.83/608.08, supporting its predictive contribution within these two checkpoints.
- Extraction MAE worsened modestly (36.1 to 39.3 tokens in split 440); improvements are not uniform across every family.

All task-text features derive from the existing frozen training-input export. Candidate tokenizer estimates remain excluded. No realized token counters, request counts, termination, quality, seed, latency, task IDs, or group IDs are predictors. Source and task family remain predictors; transfer to other source distributions is untested. Public tests are part of the declared task input; hidden references are not used.

## Budget development experiment

A separate five-fold experiment split each outer training fold into model-fit groups (64% of all training groups) and internal calibration groups (16%), then evaluated on the outer 20%. Thus no evaluated group trained or calibrated its own budget. Each inner calibration score is the maximum normalized upper-bound residual across all observed model/seed runs of a task group. We used the finite-sample 90% order statistic and never reduced a budget below the learned quantile.

| Budget method | Individual-run coverage | All-observed-runs task-group coverage | Mean budget |
|---|---:|---:|---:|
| Learned q90, uncalibrated | 81.54% | 73.38% | 2,277 tokens |
| Group-calibrated development budget | 94.00% | 89.62% | 2,596 tokens |

Calibration increases mean budgets by about 14%. Mean exceedance decreases from 142 to 117 tokens per run, but q90 pinball loss worsens from 189.46 to 196.45. Group coverage varies by family: 83.57% for ideation/tabular and 95.43% for grounded QA. This is empirical training-only evidence, not a claim of family-conditional or deployment coverage. A nominal guarantee would require appropriate exchangeability assumptions, which the purposive deterministic study design does not establish.

The saved budget model was subsequently fitted on all training data and is explicitly UNCALIBRATED. The fold-specific development correction factors must not be attached to that all-training model. A fresh correction needs the official calibration inputs and outcomes after the procedure is locked. The existing official calibration outcomes were not used in any experiment, model fit, threshold choice, or candidate selection.

## Data integrity and dependence

8,396 complete training runs represent 2,100 task groups. All 4,198 complete same-task/same-model seed pairs have exactly identical cumulative token totals. This does not prove full trajectories are identical. Seeds are repeated configurations and do not increase the number of independent task groups. Grouped splitting and task-cluster resampling preserve this dependence. Four missing training outcomes remain missing and are never imputed. Predictions are conditional on complete telemetry and the study's controller, source selection, and runtime.

Checks passed: source export SHA-256 hashes, unique training IDs, token arithmetic, training-only fitting, disjoint task folds, ignored post-execution fields in feature extraction, serialized-model prediction round-trip, and the prediction CLI on two tasks for both models. The official confirmation outcomes were neither accessed nor exported. The Roar export utility below has been syntax-checked but cannot be integration-tested without the frozen Roar directory.

## Files and reproduction

Set `TRUECOST_ML_ROOT` to the existing ml_starter folder containing `data/20261020T021459Z` and `data/ml_feature_provenance/training_task_inputs.json`. The default is the local Downloads/ml_starter path used for these runs. Run from this extracted folder with Python 3.12, numpy, pandas, scipy, scikit-learn 1.8.0, and joblib.

```bash
export TRUECOST_ML_ROOT=/path/to/ml_starter
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 improve_predictions.py --seed 440 --out results_seed440
python3 improve_predictions.py --seed 917 --out results_seed917
python3 test_specialists.py
python3 check_ablations.py
python3 check_budget_bounds.py
python3 finalize_development.py
```

`selected_models/` contains point.joblib, budget_uncalibrated.joblib, and a development configuration/provenance record. Load only trusted joblib files. `predict_tasks.py` takes the same frozen pre-run JSON schema and returns both model predictions per task; its q90 column is explicitly uncalibrated. Candidate results, out-of-fold predictions, family diagnostics, logs, and group-budget results are retained in the accompanying directories. Original user data and source attachments were not modified, and raw task texts/outcomes are not duplicated in this deliverable.

## Exact next input needed

Provide `calibration_task_inputs.json` for the 450 official calibration task groups, with the same pre-run descriptor schema as the existing training input export. Only pre-run inputs are needed from Roar; the audited calibration measurements are already available locally. Keep confirmation outcomes sealed.

The included exporter verifies the known research-manifest SHA-256 and frozen-tasks hash, selects only calibration assignments, and reads no outcome files. Copy `export_calibration_inputs.py` to the collection project on Roar and run there:

```bash
python3 export_calibration_inputs.py --frozen ./frozen --out calibration_task_inputs.json
```

Return the resulting JSON and its `.provenance.json` companion. The exporter creates new files exclusively and stops on a different manifest. After obtaining these inputs, lock the calibration/evaluation specification, fit the official task-group correction without selecting a new point predictor on calibration outcomes, and evaluate once on confirmation under the fixed procedure.
