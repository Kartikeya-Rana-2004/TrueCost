# TrueCost completion feedback — first implementation

This module records pre-execution predictions, links one execution outcome to each prediction ID, calculates post-completion errors and exports candidate training records. Python 3.10+; the ledger itself uses only the standard library. Data persists in a local SQLite file.

## Try the complete workflow

From this directory:

```bash
python3 feedback.py --db feedback.sqlite predict prediction.example.json
# Copy the returned prediction_id into the next command after execution:
python3 feedback.py --db feedback.sqlite complete completion.example.json --prediction-id YOUR_ID
python3 feedback.py --db feedback.sqlite report
python3 feedback.py --db feedback.sqlite export > candidate_feedback.json
python3 -m unittest discover -v
```

The example files contain synthetic values, not measured project outcomes. Register the prediction before executing the task. Preserve its returned ID in the runner's execution record. If the task is executed again, register a new prediction/execution pair.

## Use the existing ML predictor

Use the same Python environment as the current ML code (including its pinned ML dependencies):

```bash
python3 predict_and_record.py --ml-dir ../ml_work --tasks /path/to/new_task_inputs.json --db feedback.sqlite --controller-version six-request-v1
```

This calls the existing `predict_tasks.py`, records its outputs for both supported models, and identifies the point model by SHA-256. It returns IDs keyed by task and model. Input tasks must follow the existing predictor's input format. It does not execute tasks. Record completions only for models actually executed; other predictions remain pending. Use only new tasks with this adapter; historical calibration/confirmation runs must be registered with their correct partition using the ledger API. Keep the calibration files and ML environment with the frozen predictor.

## Integration API

```python
from feedback import Ledger
ledger = Ledger('feedback.sqlite')
prediction_id = ledger.predict(prediction_dict)
# Execute using the existing runner, then map its observed counters:
ledger.complete(prediction_id, completion_dict)
report = ledger.report()   # JSON-ready data for dashboard cards and run detail
candidate = ledger.export()
```

Token counters must cover the entire agent execution, including all requests and retries. They must come from measured telemetry; do not infer actual totals from the prediction. Unknown counters remain null. `telemetry_complete=true` requires nonnegative integer input/output/total tokens with matching arithmetic. Preserve failed and step-limited runs with complete telemetry: resource use remains meaningful even when the task fails. Interrupted runs are analyzed if fully measured but excluded from candidate exports.

## Analysis and training boundaries

Reports include pending runs, incomplete telemetry, MAE, RMSE, signed bias (prediction minus actual), percentage error (null for actual zero), budget coverage and per-run exceedance flags. Metrics are also separated by model, model version, controller version and partition. Overall metrics pool these strata and should not be used as an apples-to-apples model comparison. Missing budgets are excluded from coverage denominators. Optional requests, retries, tool calls, latency and task success remain available in run details.

The export excludes every group that is marked training, development, calibration or confirmation anywhere in this ledger. This requires correct group identities and registration of protected groups; it cannot detect overlap with external datasets that were never registered. Before training, compare exported group IDs against the authoritative original manifests. Completed-run fields are outcomes and diagnostics, never pre-execution predictors. The export is a staging dataset, not a training feature matrix.

No automatic retraining or model promotion is implemented. Future candidates need group-disjoint evaluation and a new untouched evaluation set; the original confirmation set has already been evaluated. No cost estimates are invented for locally hosted models. Dollar cost requires an explicit pricing/accounting method. Reports describe observed associations, not causal attribution.

## Current scope

Implemented: persistent ledger, validation, post-completion metrics, JSON reporting, candidate export and a frozen-predictor adapter. The adapter requires the separate existing ML folder. Core tests run without it. No task runner hooks, web interface, authentication, automatic telemetry ingestion or retraining scheduler are included yet. The next integration step is to call `complete` from the actual runner's finalization path, including failure handling, then connect the report to the dashboard.

Prediction and completion records are append-only through this API; duplicates are rejected rather than silently overwriting prior evidence. Keep the database local and back it up. Task text is not required or stored by default.
