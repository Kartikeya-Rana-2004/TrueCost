"""Load nebius/SWE-agent-trajectories (primary dataset) from Hugging Face."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_TRAJECTORIES, TRAJECTORIES_PARQUET
from scripts.utils import log, save_df, normalize_instance_id_col


def main():
    from datasets import load_dataset

    log(f"loading {HF_TRAJECTORIES} ...")
    ds = load_dataset(HF_TRAJECTORIES)

    # Use the largest available split (usually 'train')
    split_name = max(ds.keys(), key=lambda k: len(ds[k]))
    df = ds[split_name].to_pandas()
    log(f"split '{split_name}': {len(df):,} rows, columns: {list(df.columns)}")

    df = normalize_instance_id_col(df, ["instance_id", "task_id", "id"])
    if "instance_id" not in df.columns:
        log("WARNING: no obvious instance_id column found — inspect columns above "
            "and update normalize_instance_id_col candidates.")

    df["source_dataset"] = "trajectories"
    save_df(df, TRAJECTORIES_PARQUET)


if __name__ == "__main__":
    main()
