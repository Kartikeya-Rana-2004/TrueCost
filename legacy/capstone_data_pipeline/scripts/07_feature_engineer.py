"""Build the model-ready dataset: one row per priced agent trajectory, with
a unified outcome, a within-source treatment variable, and covariates
(including a repo-level difficulty proxy borrowed from swe_smith).

Design decisions (see chat for rationale):
  - Only `trajectories` and `swe_bench_experiments` rows have a real cost
    outcome, so only they become modeling rows. `swe_smith` has no agent
    run / no cost — it contributes a covariate, not rows.
  - `treatment` is NOT unified across sources: trajectories' treatment is
    Llama model size (8b/70b/405b); swe_bench_experiments' treatment is the
    raw model/harness string. These are different vocabularies measuring
    different things (size vs. identity) — collapsing them would imply a
    comparability that doesn't exist. Use `source_dataset` to stratify any
    cross-source analysis; within a stratum, `treatment` is directly usable
    for T-learner/X-learner/Double ML.
  - `repo_avg_test_count`: for each repo, the mean of swe_smith's
    FAIL_TO_PASS_num_items + PASS_TO_PASS_num_items across all swe_smith
    instances in that repo. This is a repo-level difficulty signal applied
    to trajectories/experiments rows for the SAME repo (not the same task) —
    a coarser but real difficulty covariate given no shared task IDs exist.

Output: data/processed/model_ready_dataset.parquet
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMBINED_PARQUET, PROCESSED_DIR
from scripts.utils import log, fail

MODEL_READY_PARQUET = os.path.join(PROCESSED_DIR, "model_ready_dataset.parquet")


def extract_model_size_b(model_name):
    """swe-agent-llama-70b -> 70 (billions of params), as a numeric treatment level."""
    if not isinstance(model_name, str):
        return None
    for size in (405, 70, 8):
        if f"{size}b" in model_name.lower():
            return size
    return None


def build_repo_difficulty(df):
    """One row per repo: mean total test count from swe_smith instances there."""
    smith = df[df["source_dataset"] == "swe_smith"].copy()
    if smith.empty or "repo" not in smith.columns:
        log("WARNING: no swe_smith rows with repo — repo_avg_test_count will be all-null")
        return pd.DataFrame(columns=["repo", "repo_avg_test_count", "repo_swe_smith_task_count"])

    fail_col = "swe_smith__FAIL_TO_PASS_num_items"
    pass_col = "swe_smith__PASS_TO_PASS_num_items"
    smith["total_tests"] = smith.get(fail_col, 0).fillna(0) + smith.get(pass_col, 0).fillna(0)

    agg = (
        smith.groupby("repo")["total_tests"]
        .agg(repo_avg_test_count="mean", repo_swe_smith_task_count="count")
        .reset_index()
    )
    log(f"built repo difficulty lookup: {len(agg):,} repos from swe_smith")
    return agg


def build_modeling_rows(df):
    """Unify trajectories + swe_bench_experiments into one outcome/treatment schema."""
    rows = []

    if "trajectories__estimated_cost_usd" in df.columns:
        traj = df[df["source_dataset"] == "trajectories"].copy()
        traj["cost_usd"] = traj["trajectories__estimated_cost_usd"]
        traj["treatment"] = traj["trajectories__model_name"]
        traj["treatment_size_b"] = traj["trajectories__model_name"].map(extract_model_size_b)
        traj["input_tokens"] = traj.get("trajectories__input_tokens")
        traj["output_tokens"] = traj.get("trajectories__output_tokens")
        traj["trajectory_num_turns"] = traj.get("trajectories__trajectory_num_items")
        traj["exit_status"] = traj.get("trajectories__exit_status")
        rows.append(traj)
    else:
        log("WARNING: trajectories__estimated_cost_usd not found — run "
            "06_estimate_trajectory_costs.py and 05_combine_datasets.py first")

    if "swe_bench_experiments__cost" in df.columns:
        exp = df[df["source_dataset"] == "swe_bench_experiments"].copy()
        exp["cost_usd"] = exp["swe_bench_experiments__cost"]
        exp["treatment"] = exp["swe_bench_experiments__model"]
        exp["treatment_size_b"] = None  # not size-comparable to Llama buckets
        exp["input_tokens"] = None
        exp["output_tokens"] = None
        exp["trajectory_num_turns"] = exp.get("swe_bench_experiments__messages_num_items")
        exp["exit_status"] = None
        rows.append(exp)

    if not rows:
        fail("no source has a cost outcome — nothing to build. Run steps 01-06 first.")

    keep_cols = [
        "instance_id", "repo", "source_dataset", "cost_usd", "treatment",
        "treatment_size_b", "input_tokens", "output_tokens",
        "trajectory_num_turns", "exit_status",
    ]
    stacked = pd.concat([r[keep_cols] for r in rows], ignore_index=True)
    return stacked


def main():
    if not os.path.exists(COMBINED_PARQUET):
        fail(f"{COMBINED_PARQUET} not found — run 05_combine_datasets.py first")

    df = pd.read_parquet(COMBINED_PARQUET)

    modeling = build_modeling_rows(df)
    log(f"modeling rows before dropping missing cost: {len(modeling):,}")
    modeling = modeling.dropna(subset=["cost_usd"])
    log(f"modeling rows with a valid cost outcome: {len(modeling):,}")

    repo_difficulty = build_repo_difficulty(df)
    modeling = modeling.merge(repo_difficulty, on="repo", how="left")
    n_matched = modeling["repo_avg_test_count"].notna().sum()
    log(f"repo_avg_test_count matched for {n_matched:,} / {len(modeling):,} rows "
        f"({100 * n_matched / len(modeling):.1f}%) — unmatched rows are repos "
        f"swe_smith has no instances for")

    log(f"\nfinal shape: {modeling.shape[0]:,} rows x {modeling.shape[1]} columns")
    log(f"rows per source_dataset:\n{modeling['source_dataset'].value_counts()}")
    log(f"\ncost_usd by source_dataset:\n{modeling.groupby('source_dataset')['cost_usd'].describe()}")

    modeling.to_parquet(MODEL_READY_PARQUET, index=False)
    log(f"\nwrote {MODEL_READY_PARQUET} ({os.path.getsize(MODEL_READY_PARQUET) / 1e6:,.1f} MB)")


if __name__ == "__main__":
    main()
