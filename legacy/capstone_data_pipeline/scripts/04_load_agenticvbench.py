"""Load AgenticVBench (out-of-domain generalizability source).

IMPORTANT: AgenticVBench does not publish per-task trajectories with cost/
token data — only aggregate leaderboard scores (accuracy per model x
harness) and the task definitions themselves. This script pulls what's
actually public:

  1. Leaderboard table (agenticvbench.com/leaderboard) -> scores per
     model x harness, no cost data.
  2. Task metadata (GitHub repo) -> task family, instructions, not
     agent-run data.

For real trajectory-level cost data comparable to the SWE-agent sources,
you must run the benchmark yourself via Harbor (see run_harbor_benchmark.py
in this same folder) — that costs real API spend, order of $0.10-$2/task.
"""
import glob
import io
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENTICVBENCH_DIR, AGENTICVBENCH_PARQUET
from scripts.utils import log, save_df, fail

LEADERBOARD_URL = "https://agenticvbench.com/leaderboard"
GITHUB_TASKS_REPO = "https://github.com/PhiloLabs/agentic-vbench.git"
TASKS_CLONE_DIR = os.path.join(AGENTICVBENCH_DIR, "agentic-vbench-repo")


def fetch_leaderboard() -> pd.DataFrame:
    import requests

    log(f"fetching leaderboard from {LEADERBOARD_URL} ...")
    resp = requests.get(LEADERBOARD_URL, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        fail("no <table> found on the leaderboard page — site layout may have changed")

    df = max(tables, key=len)  # the results table is the largest one on the page
    df["source_dataset"] = "agenticvbench_leaderboard"
    log(f"leaderboard: {len(df):,} rows, columns: {list(df.columns)}")
    return df


def clone_task_repo():
    if os.path.isdir(TASKS_CLONE_DIR):
        log(f"task repo already cloned at {TASKS_CLONE_DIR}, pulling latest")
        subprocess.run(["git", "-C", TASKS_CLONE_DIR, "pull"], check=True)
    else:
        log(f"cloning {GITHUB_TASKS_REPO} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_TASKS_REPO, TASKS_CLONE_DIR],
            check=True,
        )


def load_task_metadata() -> pd.DataFrame:
    """Walk tasks/<family>/<task>/ for instruction.md files -> one row per task."""
    rows = []
    tasks_root = os.path.join(TASKS_CLONE_DIR, "tasks")
    for instr_path in glob.glob(os.path.join(tasks_root, "*", "*", "steps", "solve", "instruction.md")):
        parts = instr_path.split(os.sep)
        # .../tasks/<family>/<task>/steps/solve/instruction.md
        family = parts[-5]
        task_name = parts[-4]
        with open(instr_path, encoding="utf-8", errors="ignore") as f:
            instruction = f.read()
        rows.append({
            "task_family": family,
            "task_name": task_name,
            "instruction_length_chars": len(instruction),
        })
    df = pd.DataFrame(rows)
    df["source_dataset"] = "agenticvbench_tasks"
    log(f"task metadata: {len(df):,} rows")
    return df


def main():
    os.makedirs(AGENTICVBENCH_DIR, exist_ok=True)

    leaderboard_df = fetch_leaderboard()

    try:
        clone_task_repo()
        tasks_df = load_task_metadata()
    except subprocess.CalledProcessError as e:
        log(f"WARNING: could not clone task repo ({e}) — proceeding with leaderboard only")
        tasks_df = pd.DataFrame()

    leaderboard_path = os.path.join(AGENTICVBENCH_DIR, "leaderboard.parquet")
    save_df(leaderboard_df, leaderboard_path)

    if len(tasks_df):
        tasks_path = os.path.join(AGENTICVBENCH_DIR, "task_metadata.parquet")
        save_df(tasks_df, tasks_path)

    # No instance_id-comparable join key exists between this source and the
    # SWE-agent sources (this is model/harness-level, not per-instance-trial-level
    # in a way that maps to SWE-bench instance_id). Save as its own standalone
    # parquet rather than forcing it into 05_combine_datasets.py's join.
    save_df(leaderboard_df, AGENTICVBENCH_PARQUET)
    log("NOTE: no instance_id column — this source will be excluded from the "
        "instance_id join in 05_combine_datasets.py and reported separately.")


if __name__ == "__main__":
    main()
