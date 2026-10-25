# TrueCost frozen confirmation results

The preselected token-usage predictor and calibrated upper-budget method have completed their one-time confirmation evaluation on 450 unseen task groups comprising 1,800 complete runs. No model was refitted or tuned on confirmation results. All previously locked model, feature-code, calibration-parameter, protocol, and evaluator hashes remained unchanged.

## Final confirmation results

| Metric | Value |
|---|---:|
| Mean absolute error | 323.78 tokens |
| Root mean squared error | 1,404.38 tokens |
| R² | 0.7052 |
| Mean signed error, prediction minus actual | −142.70 tokens |
| 90th percentile absolute error | 767.83 tokens |
| Calibrated upper-budget coverage per run | 95.33% (1,716/1,800) |
| Joint coverage of all four runs per task group | 91.56% (412/450) |
| Mean upper budget | 2,592.85 tokens |
| Mean budget exceedance across all runs | 79.94 tokens |
| q90 pinball loss of calibrated budget | 157.98 tokens |

Joint group coverage exceeded the prespecified 90% target in this confirmation sample. This is an empirical result for the collected workload, rather than a universal or family-conditional guarantee. The point predictor still underestimates total usage on average. A lower MAE here than in development cross-validation describes different partitions and is not a measured additional improvement from calibration: calibration left the point predictor unchanged.

## Expensive tasks remain a limitation

The tail was fixed in advance at actual usage of at least 5,241 tokens, the training-set 90th percentile. Among 176 confirmation runs meeting that threshold, MAE was 1,414.09 tokens and mean signed error was −1,313.43. These estimates concern the actual expensive-run subset; they are not a mechanism for identifying such tasks before execution. The larger RMSE than MAE shows that a small number of large errors remain important.

## Family and model differences

| Task family | Task groups | MAE, tokens | All-four-runs group coverage |
|---|---:|---:|---:|
| Classification | 60 | 49.16 | 90.00% |
| Coding | 30 | 849.84 | 83.33% |
| Extraction | 60 | 28.00 | 96.67% |
| Grounded QA | 75 | 30.31 | 98.67% |
| Ideation | 30 | 86.42 | 73.33% |
| Math | 75 | 565.07 | 92.00% |
| Summarization | 60 | 453.75 | 98.33% |
| Tabular reasoning | 60 | 685.09 | 85.00% |

The global correction does not ensure 90% coverage within every family. Ideation, coding, and tabular reasoning fall below that level. Family samples, particularly the 30-group coding and ideation subsets, are small. Summarization has MAE 453.75 but RMSE 3,123.61, indicating large outliers despite strong budget coverage. Classification and ideation have weak within-family R² (0.009 and −0.166); overall R² should not be used to imply equal skill in every family.

| Model | Runs | MAE, tokens | RMSE, tokens | Mean signed error |
|---|---:|---:|---:|---:|
| qwen2.5-coder:14b | 900 | 170.96 | 355.45 | −36.04 |
| qwen2.5-coder:7b | 900 | 476.60 | 1,954.03 | −249.37 |

## Fixed method

The point predictor is histogram gradient boosting fitted on the 8,396 complete training runs with absolute-error loss, 300 iterations, 15 maximum leaf nodes, minimum leaf size 25, L2 regularization 10, no early stopping, and random seed 440. Inputs are pre-execution task family, source, size band, model identity, and text-derived descriptors. Realized requests, telemetry, termination, quality, latency, seed, and task/group identifiers are excluded as predictive inputs.

A separate frozen quantile model estimates q90 total usage. Calibration used the maximum normalized upper-budget residual across all four model/seed runs of each of 450 calibration tasks. The 406th ordered group score produced a correction of 0.13695943800781887. The returned upper budget is `ceil(max(1, q90_prediction) * 1.136959438007819)`. The point prediction is unchanged and clamped at zero. This is a one-sided joint group budget, not a two-sided prediction interval.

## Verification and evaluation boundary

The received archive's task, run-table, and audit-summary hashes matched its provenance. Its research-manifest hash matched the original campaign. The Roar strict-audit receipt recorded complete evidence for all four confirmation shards, each with 450/450 completed assignments and zero pending. That receipt attests to the original raw-evidence audit; local export checks do not independently reproduce an audit of underlying raw traces, which are not in the ZIP.

All 1,800 run IDs matched the previously exported confirmation assignment dispositions. Metadata, four-assignment group coverage, complete telemetry, and token arithmetic passed. Confirmation task groups were disjoint from both training and calibration. All locked artifacts matched their recorded SHA-256 values before and after evaluation. The evaluator ran successfully once and wrote its original receipt; verification afterward used saved predictions and did not rerun inference or fit models.

The final evaluation receipt and machine-readable metrics are in `confirmation_evaluation/`. Prior readiness records intentionally retain their historical status; `confirmation_evaluation/final_status.json` records completion. The original source export is retained separately as the user-uploaded `TrueCost_Confirmation_Evaluation.zip`.

## Interpretation and next research cycle

These results support pre-execution total-token prediction and upper budgeting for the study's particular two model checkpoints, eight task families, source distributions, and bounded six-request controller. They do not establish dollar cost, energy use, general task correctness, or a causal decomposition of architecture effects. Four incomplete training outcomes were not imputed. All final confirmation groups are fully measured.

The data were purposively selected and deterministically partitioned. Repeated seed configurations are correlated; task groups, not individual runs, are the relevant unit for joint calibration. Claims of nominal distribution-free coverage would require suitable exchangeability assumptions that this design does not establish.

Future work should investigate costly-run mechanisms and the lower-coverage families. Any candidate feature, model, or family correction informed by these confirmation results belongs to a new development cycle. Preserve this result as the final evaluation of the frozen version and collect fresh independent evaluation tasks before claiming improvements. No new user decision is needed to preserve the completed results; new collection would require access to Roar and a defined new sample plan.

## Package contents

The package includes the frozen models, calibration parameters and protocol, reproducible experiment scripts, previous development/calibration results, the confirmation metrics and predictions, evaluation receipt, final consistency checks, and archive hashes. Frozen task texts are not duplicated; reproducing the export-based final evaluation also requires the already supplied calibration/confirmation input archives and audited assignment data. Do not refit the selected models and present another evaluation on these same confirmation tasks as an untouched test.
