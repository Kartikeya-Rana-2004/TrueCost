"""Combine trajectories, SWE-smith, SWE-bench/experiments, and AgenticVBench
into one dataset.

These sources do NOT share task instances (confirmed empirically: 0%
instance_id overlap between trajectories, swe_smith, and
swe_bench_experiments — they draw from different repo populations, and
swe_smith's instance_ids follow a different synthetic-task scheme entirely).
So this does NOT join them into a wide per-instance table. Instead it
STACKS them into one long table: one row per agent trajectory/run, with
`source_dataset` identifying which benchmark it came from, `repo`
extracted where possible for repo-level grouping, and every other column
kept but namespaced by source to avoid accidental collisions between
same-named columns that mean different things in different sources.

AgenticVBench has no instance_id at all (it's leaderboard-level, not
per-instance) and is written to its own separate file rather than
stacked in — it's a different unit of observation (model x harness
score, not a single task trajectory).
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAJECTORIES_PARQUET,
    SWE_SMITH_PARQUET,
    EXPERIMENTS_PARQUET,
    AGENTICVBENCH_PARQUET,
    PROCESSED_DIR,
    COMBINED_PARQUET,
    COMBINED_CSV,
    JOIN_KEY,
)
from scripts.utils import log, fail

STACKABLE_SOURCES = {
    "trajectories": TRAJECTORIES_PARQUET,
    "swe_smith": SWE_SMITH_PARQUET,
    "swe_bench_experiments": EXPERIMENTS_PARQUET,
}

AGENTICVBENCH_STANDALONE_PATH = os.path.join(PROCESSED_DIR, "agenticvbench_standalone.parquet")

# Columns that can be MB-scale per cell (full conversation transcripts, eval
# logs, message arrays). Keeping these in the combined table is what
# produces multi-GB output — the raw text is still in data/raw/<source>.parquet
# for anyone who needs it. Here they're replaced with cheap derived features
# (character length, item count) that are actually usable as model inputs.
HEAVY_COLUMNS = {
    "trajectories": ["trajectory", "eval_logs", "generated_patch"],
    "swe_bench_experiments": ["messages", "info", "preds", "golds"],
    "swe_smith": ["FAIL_TO_PASS", "PASS_TO_PASS"],
}


def _json_array_len(v):
    """If v looks like a JSON-serialized list, return its item count."""
    if not isinstance(v, str) or not v.startswith("["):
        return None
    try:
        return len(json.loads(v))
    except (json.JSONDecodeError, ValueError):
        return None


def summarize_and_drop_heavy(df, name):
    for col in HEAVY_COLUMNS.get(name, []):
        if col not in df.columns:
            continue
        df[f"{col}_chars"] = df[col].map(lambda v: len(v) if isinstance(v, str) else 0)
        df[f"{col}_num_items"] = df[col].map(_json_array_len)
        df = df.drop(columns=[col])
        log(f"{name}: replaced heavy column '{col}' with '{col}_chars'/'{col}_num_items'")
    return df


def load_available(sources):
    dfs = {}
    for name, path in sources.items():
        if os.path.exists(path):
            dfs[name] = pd.read_parquet(path)
            log(f"{name}: {len(dfs[name]):,} rows")
        else:
            log(f"{name}: not found at {path} — run its load script first, skipping")
    return dfs


def extract_repo(instance_id):
    """owner__repo-1234 -> owner/repo ; owner__repo.<hash>.combine_file__x -> owner/repo"""
    if not isinstance(instance_id, str) or "__" not in instance_id:
        return None
    owner, rest = instance_id.split("__", 1)
    m = re.match(r"^(.+)-\d+$", rest)
    repo_name = m.group(1) if m else rest.split(".")[0]
    return f"{owner}/{repo_name}"


def report_overlap(dfs):
    id_sets, repo_sets = {}, {}
    for name, df in dfs.items():
        if JOIN_KEY not in df.columns:
            log(f"{name}: no '{JOIN_KEY}' column")
            continue
        ids = df[JOIN_KEY].dropna().astype(str)
        id_sets[name] = set(ids)
        repo_sets[name] = set(ids.map(extract_repo).dropna())

    names = list(id_sets.keys())
    log("--- instance_id overlap (exact task match) ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = len(id_sets[a] & id_sets[b])
            union = len(id_sets[a] | id_sets[b]) or 1
            log(f"  {a} <-> {b}: {inter} shared / {union} union ({100 * inter / union:.1f}%)")

    log("--- repo overlap (same repo, possibly different task/issue) ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = len(repo_sets[a] & repo_sets[b])
            union = len(repo_sets[a] | repo_sets[b]) or 1
            log(f"  {a} <-> {b}: {inter} shared / {union} union ({100 * inter / union:.1f}%)")


def build_stacked(dfs):
    frames = []
    for name, df in dfs.items():
        df = df.copy()
        if JOIN_KEY in df.columns:
            df["repo"] = df[JOIN_KEY].astype(str).map(extract_repo)
        df = summarize_and_drop_heavy(df, name)
        # Namespace every column except the small set of shared/common ones,
        # so e.g. swe_bench_experiments' "model" and trajectories' "model_name"
        # never silently collide or get misread as the same field.
        common = {JOIN_KEY, "repo", "source_dataset"}
        df = df.rename(columns={c: f"{name}__{c}" for c in df.columns if c not in common})
        frames.append(df)

    stacked = pd.concat(frames, ignore_index=True, sort=False)
    return stacked


def main():
    dfs = load_available(STACKABLE_SOURCES)
    if not dfs:
        fail("no stackable source parquet files found — run 01/02/03 load scripts first")

    report_overlap(dfs)

    stacked = build_stacked(dfs)
    log(f"stacked dataset: {len(stacked):,} rows, {len(stacked.columns)} columns")
    log(f"rows per source: {stacked['source_dataset'].value_counts().to_dict()}")

    stacked.to_parquet(COMBINED_PARQUET, index=False)
    stacked.to_csv(COMBINED_CSV, index=False)
    for p in (COMBINED_PARQUET, COMBINED_CSV):
        log(f"wrote {p} ({os.path.getsize(p) / 1e6:,.1f} MB)")

    if os.path.exists(AGENTICVBENCH_PARQUET):
        avb = pd.read_parquet(AGENTICVBENCH_PARQUET)
        avb.to_parquet(AGENTICVBENCH_STANDALONE_PATH, index=False)
        log(f"AgenticVBench has no instance_id (leaderboard-level, not per-task) — "
            f"kept separate at {AGENTICVBENCH_STANDALONE_PATH}, not stacked in")


if __name__ == "__main__":
    main()
