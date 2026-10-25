# Causal Cost Attribution — Data Pipeline

Loads and combines the four capstone datasets into one joined dataset for
causal ML modeling (T-learner / X-learner / Double ML).

## Requires internet access (Colab/Kaggle)

This must run in an environment with open internet (Hugging Face, GitHub,
optionally AWS S3) — not a sandboxed environment.

## Setup

```bash
pip install -r requirements.txt
```

If pulling `SWE-bench/experiments` from S3, set AWS credentials as env vars
first (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or via `aws configure`.

## Run order

```bash
python scripts/run_all.py
```

or step by step:

```bash
python scripts/01_load_trajectories.py
python scripts/02_load_swe_smith.py
python scripts/03_load_swe_bench_experiments.py
python scripts/04_load_agenticvbench.py
python scripts/05_combine_datasets.py
```

Each `0X_load_*.py` script writes a cleaned parquet file to `data/raw/`.
`05_combine_datasets.py` **stacks** trajectories, SWE-smith, and
SWE-bench/experiments into one long dataset (`data/processed/combined_dataset.parquet`)
— one row per agent trajectory, `source_dataset` marking origin. It does
**not** join them on `instance_id`: empirically, these three sources cover
disjoint repo/task populations (confirmed 0% instance-level overlap), so a
per-instance join isn't meaningful. AgenticVBench (leaderboard-level, not
per-task) is written separately to `agenticvbench_standalone.parquet`.

## AgenticVBench

No manual download needed — `04_load_agenticvbench.py` scrapes the public
leaderboard (aggregate model x harness scores, no cost data) and clones the
task-definition repo automatically. It has no `instance_id` column, so it's
excluded from the join in `05_combine_datasets.py` and saved standalone.

Real per-task trajectories with cost/token data are **not published** for
this source — only generatable by actually running the 100 tasks yourself
via the Harbor framework (`scripts/run_harbor_benchmark.py`, optional, costs
real API spend: ~$0.10–$2/task, ~10 min/task wall clock).

## instance_id overlap — resolved

Confirmed empirically: `nebius/SWE-agent-trajectories`, `SWE-bench/SWE-smith`,
and `SWE-bench/experiments` share **0% of instance_ids** and are drawn from
different repo populations — this is a real content mismatch, not a
formatting bug. `05_combine_datasets.py` therefore stacks them (rows) rather
than joining them (columns); see `scripts/diagnose_ids.py` if you want to
re-inspect sample IDs per source.
