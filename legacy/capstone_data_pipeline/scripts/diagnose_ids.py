"""Diagnostic: print sample instance_id values (and all column names) from
each raw parquet so ID-format mismatches can actually be seen and fixed.
Run this after the 0X_load_*.py scripts, before 05_combine_datasets.py.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAJECTORIES_PARQUET,
    SWE_SMITH_PARQUET,
    EXPERIMENTS_PARQUET,
    JOIN_KEY,
)

SOURCES = {
    "trajectories": TRAJECTORIES_PARQUET,
    "swe_smith": SWE_SMITH_PARQUET,
    "swe_bench_experiments": EXPERIMENTS_PARQUET,
}

for name, path in SOURCES.items():
    print(f"\n=== {name} ===")
    if not os.path.exists(path):
        print(f"  (not found at {path})")
        continue
    df = pd.read_parquet(path)
    print(f"  columns: {list(df.columns)}")
    if JOIN_KEY in df.columns:
        sample = df[JOIN_KEY].dropna().astype(str).unique()[:10]
        print(f"  sample {JOIN_KEY} values ({df[JOIN_KEY].nunique()} unique):")
        for v in sample:
            print(f"    {v!r}")
    else:
        print(f"  no '{JOIN_KEY}' column present")
