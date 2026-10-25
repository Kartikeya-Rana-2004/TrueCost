# TrueCost V2: first development cycle

V1.0.0 is preserved in `../releases/v1.0.0/`, with a copy of the final deliverable, expanded artifacts, SHA-256 manifest, and read-only snapshot files. Its final confirmation performance remains MAE 323.78 tokens and joint task-group budget coverage 91.56%. V2 has not replaced V1.

This cycle completed training-trace diagnostics, feature ablations, token-based mixture tests, request-limit/output-ceiling risk forecasts, and request-limit expert routing. All experiments used only the original 8,396 training runs from 2,100 task groups. Neither calibration nor confirmation outcomes informed these fits or comparisons. Prior confirmation observations motivated the overall V2 research direction, so any future V2 result on that old confirmation set is development evidence, not a fresh final test.

## What the traces showed

- 912/8,396 runs (10.86%) reached the six-request cap. They accounted for 59.67% of total absolute OOF prediction error. Their MAE was 1,977.52 tokens versus 162.87 for other runs.
- The worst 84 runs (approximately 1%) accounted for 33.33% of absolute error. These are correlated run observations rather than 84 independent tasks.
- 66 runs across 30 task groups had at least one request with output at the 1,024-token ceiling. Their MAE was 11,258.66 tokens; they contributed approximately 24.59% of total absolute error. These ceiling events overlap with request-cap cases and must not be added as disjoint contributions.
- The largest-error summarization traces included repeated `read_context` actions and model responses that could not be decoded as JSON by the diagnostic parser. These are observations about the collected behavior, not proof of the causal effect of JSON formatting or controller settings.
- Inputs processed after the first request can include newly retrieved context as well as repeated history. They are not a direct measure of wasted tokens.

The streaming trace reader checked report metadata, request sequence, and ledger totals against the audited training table for every admitted training run. It ignored calibration/confirmation trace contents and never extracted archive paths. These consistency checks do not replace the original strict evidence audit. Diagnostic execution fields are excluded from every pre-execution feature list.

## Matched grouped-validation comparisons

All candidates used the original V1 five-fold task-group assignments for fold seeds 440 and 917. The baseline was refitted within each fold using its unchanged configuration; its predictions match V1's saved OOF predictions. A second seed is sensitivity analysis on the same tasks, not independent evaluation. Candidate choices evolved from training diagnostics, and all results are exploratory development evidence.

| Candidate | MAE, split 440 | MAE, split 917 |
|---|---:|---:|
| V1 baseline | 359.98 | 359.44 |
| Add request-format descriptors | 359.27 | 359.06 |
| Add text/table/code structure descriptors | 361.38 | 360.10 |
| Add all new descriptors | 361.54 | 360.13 |
| High-token-risk soft expert mixture | 370.61 | 368.01 |
| High-token-risk hard expert routing | 376.46 | 369.13 |
| Request-limit-risk soft expert mixture | 385.27 | 392.79 |
| Request-limit-risk hard expert routing | 371.19 | 378.77 |

Request-format features improved MAE by only 0.71 and 0.37 tokens (about 0.20% and 0.10%). This small exploratory gain is insufficient evidence to promote a new predictor. They also slightly worsened expensive-run signed bias. Hard request-limit routing lowered tail MAE to 1,620.39/1,656.97 from the baseline's 1,691.71/1,700.01, but worsened overall MAE. No candidate improved the full set of objectives consistently.

The expert models used realized high-token or request-limit status only to label fit-fold training rows. Validation routing used predicted probabilities from a classifier trained on other task groups. The soft mixture combines conditional median estimates and is a heuristic; it is not mathematically the marginal conditional median.

Some proposed structure signals are absent or nearly absent in this dataset: code fences, inline code spans, and assert markers are zero across the frozen training input export. Public tests are represented only by a boolean in that export; we cannot infer detailed test counts from it. Better public-test features would need a verified pre-execution source, rather than reading generated solution code or hidden outcomes.

## Execution-risk findings

A request-limit classifier achieved average precision 0.931/0.930 versus prevalence 0.109. However, a smoothed family/model-rate baseline achieved 0.928/0.929 and had better Brier/log losses. Much of the apparently strong risk discrimination is therefore already captured by task family and model identity. The more complex classifier should not be presented as a major new forecasting gain.

The output-ceiling target is rare (0.786% of runs). Its classifier achieved average precision 0.074/0.061. Brier and log-loss comparisons with constant-prevalence predictions were mixed. The existing 30 affected task groups do not establish a reliable calibrated ceiling-risk model. Risk probabilities were not calibrated or deployed.

## Decision and next step

Keep V1 as the usable checkpoint. Preserve all negative comparisons. The next step is a fresh-data design focused on rare output-ceiling/formatting behavior and execution-limit cases, followed by either better pre-execution semantic descriptors or a controller-aware budgeting study. Changing the JSON policy, output limit, or request cap changes the workload definition and requires a separately versioned collection; do not modify or relabel V1 traces retroactively.

Before further point-model tuning, inventory unused source tasks on Roar and inspect the 30 training groups with ceiling events for pre-execution patterns. Detailed trace text may be used for development diagnosis, but realized behaviors cannot become features in a pre-run predictor. New task groups must be deduplicated against all prior research, smoke and pilot groups, including related variants.

A collection plan should separate targeted development tasks, fresh calibration tasks, and an untouched final test. A risk-enriched diagnostic sample should not silently become a representative performance estimate. Family-aware calibration needs enough independent groups per family, rather than treating repeated seeds as additional calibration samples. See `NEXT_COLLECTION_PLAN.md` for the staged design and decision gates. No new Roar jobs were submitted in this cycle.

## Reproduce and inspect

Dependencies are the V1 Python environment (numpy, pandas, scipy, scikit-learn 1.8.0, joblib). Scripts default to the existing `/Users/kay/Downloads/ml_starter` inputs. The raw trace script also accepts `--archive` and `--starter`; feature ablations accept `--starter`. The two risk scripts use the same default starter path, which can be edited in the V2 copies when moving machines. V1 is imported only from its preserved package. Inputs are not duplicated in the deliverable.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
python3 analyze_training_traces.py
python3 train_feature_ablations.py
python3 test_execution_risk.py
python3 evaluate_risk_baseline.py
python3 test_request_limit_experts.py
python3 verify_checkpoint_and_leakage.py
```

Run these scripts from their extracted `ml_v2` directory alongside `releases/`. The bundle retains results, folds, OOF predictions, family slices, feature-availability counts, and protocol notes. `verification.json` records unchanged V1 hashes/permissions, matched baseline predictions and folds, grouping, and a feature guard test showing that injected post-run fields do not alter extracted pre-run descriptors. V2 models were not saved or promoted; the validated V1 models remain in the preserved release.
