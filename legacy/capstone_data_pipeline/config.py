"""Central config: paths and dataset identifiers."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Hugging Face dataset IDs
HF_TRAJECTORIES = "nebius/SWE-agent-trajectories"
HF_SWE_SMITH = "SWE-bench/SWE-smith"

# SWE-bench/experiments
GITHUB_EXPERIMENTS_REPO = "https://github.com/SWE-bench/experiments.git"
EXPERIMENTS_CLONE_DIR = os.path.join(RAW_DIR, "swe_bench_experiments_repo")
# Optional: set if results also live in an S3 bucket you have access to
S3_EXPERIMENTS_BUCKET = None  # e.g. "swe-bench-experiments"
S3_EXPERIMENTS_PREFIX = ""

# AgenticVBench — manual download, no public API
AGENTICVBENCH_DIR = os.path.join(RAW_DIR, "agenticvbench_manual")

# Output parquet paths for each stage
TRAJECTORIES_PARQUET = os.path.join(RAW_DIR, "trajectories.parquet")
SWE_SMITH_PARQUET = os.path.join(RAW_DIR, "swe_smith.parquet")
EXPERIMENTS_PARQUET = os.path.join(RAW_DIR, "swe_bench_experiments.parquet")
AGENTICVBENCH_PARQUET = os.path.join(RAW_DIR, "agenticvbench.parquet")

COMBINED_PARQUET = os.path.join(PROCESSED_DIR, "combined_dataset.parquet")
COMBINED_CSV = os.path.join(PROCESSED_DIR, "combined_dataset.csv")

# Join key — verify this matches across sources before trusting the join
JOIN_KEY = "instance_id"
