"""Load SWE-bench/SWE-smith (difficulty proxy) from Hugging Face."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_SWE_SMITH, SWE_SMITH_PARQUET
from scripts.utils import log, save_df, normalize_instance_id_col


def main():
    from datasets import load_dataset

    log(f"loading {HF_SWE_SMITH} ...")
    ds = load_dataset(HF_SWE_SMITH)

    split_name = max(ds.keys(), key=lambda k: len(ds[k]))
    df = ds[split_name].to_pandas()
    log(f"split '{split_name}': {len(df):,} rows, columns: {list(df.columns)}")

    df = normalize_instance_id_col(df, ["instance_id", "task_id", "id"])
    if "instance_id" not in df.columns:
        log("WARNING: no obvious instance_id column found — inspect columns above "
            "and update normalize_instance_id_col candidates.")

    df["source_dataset"] = "swe_smith"
    save_df(df, SWE_SMITH_PARQUET)


if __name__ == "__main__":
    main()
