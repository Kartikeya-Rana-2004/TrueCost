"""Sanity-check data/processed/combined_dataset.parquet.

Run this after 05_combine_datasets.py and paste the output back for review.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMBINED_PARQUET, JOIN_KEY
from scripts.utils import log, fail

# Heavy raw-text columns that should NOT appear in the combined file —
# their presence means summarize_and_drop_heavy() didn't run or didn't match.
SHOULD_NOT_EXIST = [
    "trajectories__trajectory",
    "trajectories__eval_logs",
    "trajectories__generated_patch",
    "swe_bench_experiments__messages",
    "swe_bench_experiments__info",
    "swe_bench_experiments__preds",
    "swe_bench_experiments__golds",
]


def main():
    if not os.path.exists(COMBINED_PARQUET):
        fail(f"{COMBINED_PARQUET} not found — run 05_combine_datasets.py first")

    df = pd.read_parquet(COMBINED_PARQUET)

    log(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log(f"file size: {os.path.getsize(COMBINED_PARQUET) / 1e6:,.1f} MB")

    log("\n--- rows per source_dataset ---")
    print(df["source_dataset"].value_counts())

    log("\n--- heavy columns that should have been summarized away ---")
    leaked = [c for c in SHOULD_NOT_EXIST if c in df.columns]
    if leaked:
        log(f"PROBLEM: these raw heavy columns are still present: {leaked}")
    else:
        log("OK — none of the known heavy raw-text columns are present")

    log("\n--- largest columns by average string length (top 10) ---")
    sizes = {}
    for col in df.columns:
        if df[col].dtype == object:
            sizes[col] = df[col].astype(str).str.len().mean()
    for col, avg_len in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {col}: avg {avg_len:,.0f} chars/row")

    log(f"\n--- null counts on key columns ---")
    for col in [JOIN_KEY, "repo", "source_dataset"]:
        if col in df.columns:
            n_null = df[col].isna().sum()
            print(f"  {col}: {n_null:,} nulls / {len(df):,} rows ({100 * n_null / len(df):.1f}%)")

    log("\n--- sample rows ---")
    print(df.sample(min(3, len(df)), random_state=0).to_string())

    log("\n--- full column list ---")
    print(list(df.columns))


if __name__ == "__main__":
    main()
