"""Load SWE-bench/experiments — multi-architecture comparison data.

Clones the GitHub repo (contains per-model result JSON/JSONL files) and,
if S3_EXPERIMENTS_BUCKET is set in config.py, also pulls objects from S3.
"""
import glob
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GITHUB_EXPERIMENTS_REPO,
    EXPERIMENTS_CLONE_DIR,
    S3_EXPERIMENTS_BUCKET,
    S3_EXPERIMENTS_PREFIX,
    EXPERIMENTS_PARQUET,
)
from scripts.utils import log, save_df, normalize_instance_id_col, fail


def clone_repo():
    if os.path.isdir(EXPERIMENTS_CLONE_DIR):
        log(f"repo already cloned at {EXPERIMENTS_CLONE_DIR}, pulling latest")
        subprocess.run(["git", "-C", EXPERIMENTS_CLONE_DIR, "pull"], check=True)
    else:
        log(f"cloning {GITHUB_EXPERIMENTS_REPO} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_EXPERIMENTS_REPO, EXPERIMENTS_CLONE_DIR],
            check=True,
        )


ID_LIKE_KEYS = {"instance_id", "task_id", "id"}


def load_result_files():
    """Find and load per-instance result records (json/jsonl) from the repo.

    Only keeps records that actually have an instance/task-id-like key.
    Skips leaderboard/summary/aggregate files (repo x year x status tables,
    config files, etc.) which have no such key and would otherwise explode
    the column count with hundreds of one-off summary columns.
    """
    patterns = [
        os.path.join(EXPERIMENTS_CLONE_DIR, "**", "*.json"),
        os.path.join(EXPERIMENTS_CLONE_DIR, "**", "*.jsonl"),
    ]
    records = []
    skipped_files = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            model_name = os.path.relpath(path, EXPERIMENTS_CLONE_DIR).split(os.sep)[0]
            try:
                if path.endswith(".jsonl"):
                    with open(path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            if not (isinstance(rec, dict) and ID_LIKE_KEYS & rec.keys()):
                                continue
                            rec["_source_file"] = path
                            rec["_model_arch"] = model_name
                            records.append(rec)
                else:
                    with open(path) as f:
                        data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for rec in items:
                        if isinstance(rec, dict) and ID_LIKE_KEYS & rec.keys():
                            rec["_source_file"] = path
                            rec["_model_arch"] = model_name
                            records.append(rec)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # skip non-data files (configs, READMEs, etc.)
    return records


def maybe_pull_s3():
    if not S3_EXPERIMENTS_BUCKET:
        log("S3_EXPERIMENTS_BUCKET not set — skipping S3 pull (repo-only)")
        return []
    import boto3

    log(f"pulling from s3://{S3_EXPERIMENTS_BUCKET}/{S3_EXPERIMENTS_PREFIX} ...")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    records = []
    for page in paginator.paginate(Bucket=S3_EXPERIMENTS_BUCKET, Prefix=S3_EXPERIMENTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith((".json", ".jsonl")):
                continue
            body = s3.get_object(Bucket=S3_EXPERIMENTS_BUCKET, Key=key)["Body"].read()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            items = data if isinstance(data, list) else [data]
            for rec in items:
                rec["_source_file"] = f"s3://{S3_EXPERIMENTS_BUCKET}/{key}"
                records.append(rec)
    return records


def main():
    try:
        clone_repo()
    except subprocess.CalledProcessError as e:
        fail(f"git clone/pull failed: {e}")

    records = load_result_files()
    records += maybe_pull_s3()

    if not records:
        fail("no result records parsed from the experiments repo — inspect "
             f"{EXPERIMENTS_CLONE_DIR} manually and adjust load_result_files()")

    df = pd.DataFrame(records)
    log(f"parsed {len(df):,} rows, columns: {list(df.columns)}")

    df = normalize_instance_id_col(df, ["instance_id", "task_id", "id"])
    if "instance_id" not in df.columns:
        log("WARNING: no obvious instance_id column found — inspect columns above "
            "and update normalize_instance_id_col candidates.")

    df["source_dataset"] = "swe_bench_experiments"
    save_df(df, EXPERIMENTS_PARQUET)


if __name__ == "__main__":
    main()
