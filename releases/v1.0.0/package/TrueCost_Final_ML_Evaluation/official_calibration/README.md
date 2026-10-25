# TrueCost official calibration — October 21, 2026

The received ZIP passed its export SHA-256 and research-manifest checks. All 450 calibration task groups matched the audited data exactly, with four complete model/seed assignments per group (1,800 runs), no training overlap, and no missing token measurements. Both selected models and the feature implementation match the prior saved development deliverable. Neither model was refitted or selected on calibration outcomes.

Before inspecting calibration outcomes, `locked_analysis_protocol.json` fixed the score, 90% group coverage target, correction calculation, rounding, metrics, and confirmation evaluation plan. Calibration used one score per task group: the largest normalized excess of actual total tokens over the frozen q90 estimate across its four runs. The correction is the 406th ordered score among 450 groups, clipped at zero. The saved parameter is 0.13695943800781887.

For a new task/model configuration under the same study policy:

1. Produce the frozen point prediction, clamped at zero.
2. Produce the frozen q90 estimate `q`, floored at one token.
3. Return the upper budget `ceil(q * 1.136959438007819)`.

The point estimate is unchanged. The upper budget is a separate one-sided bound jointly calibrated over the two declared models and two seeds per task group. It is not a two-sided prediction interval or a statement of 90% coverage conditional on a specific task/family/model. New models, input distributions, controller limits, or runtimes require validation and possibly fresh calibration.

| Calibration diagnostic | Uncalibrated q90 | Calibrated upper budget |
|---|---:|---:|
| Run coverage | 83.00% | 94.67% |
| All-four-runs task-group coverage | 73.78% | 90.67% |
| Mean budget, tokens | 2,297.71 | 2,612.91 |
| Mean budget exceedance, tokens/run | 117.83 | 89.31 |
| q90 pinball loss | 161.77 | 164.76 |

The correction increases the mean budget by approximately 13.7%, improving coverage at the cost of greater budget slack. These are in-calibration fit diagnostics; 90.67% is largely determined by the chosen order statistic and ties. It is not independent evidence of future coverage. Exchangeability would be needed for nominal coverage guarantees, and this purposive deterministic dataset does not establish that assumption.

The frozen point predictor's calibration diagnostics are MAE 345.70 tokens, RMSE 1,229.40, R² 0.738, mean signed error −156.03, and 90th-percentile absolute error 813.43. Among 208 runs at or above the training-only 90th-percentile threshold of 5,241 tokens, MAE is 1,458.26 and signed bias is −1,381.42. These figures describe a different partition from development CV and do not establish a further model improvement. No calibration diagnostic was used for tuning.

`calibration_predictions.csv`, `calibration_group_scores.csv`, and `budget_diagnostics.csv` retain the results. `calibration_parameters.json` contains the applied correction and source/model hashes. `point_diagnostics.json` records point-error summaries. The new `predict_tasks.py` automatically loads these calibration parameters when present and verifies the model hashes. The budget model's original filename remains `budget_uncalibrated.joblib`: calibration is applied as a separate frozen parameter.

Checks passed: input ZIP membership and export hash; original research-manifest identity; audited-export hashes; metadata alignment; unique IDs; exact four-assignment groups; no training overlap; complete telemetry and token arithmetic; prior model/feature hashes; one-sided quantile/order-statistic tests; group-score/bound equivalence; and unchanged model files. The prediction CLI produced 900 task/model predictions that exactly matched all 1,800 calibration-run point predictions and rounded budgets after seed expansion. The original strict collection audit remains the source of telemetry admission; this script performs export consistency checks rather than re-auditing raw traces.

The confirmation set has not been read. The next required input is an audited confirmation export for the 450 confirmation task groups (1,800 complete runs) plus their pre-run task inputs and provenance. Its run IDs must match the existing confirmation assignment dispositions, and telemetry must pass the original strict evidence audit. `ready_for_confirmation.json` pins the model, feature implementation, calibration parameters, protocol, and evaluator. Evaluate exactly once under this specification and report every predeclared metric; any later changes begin a new development cycle.
