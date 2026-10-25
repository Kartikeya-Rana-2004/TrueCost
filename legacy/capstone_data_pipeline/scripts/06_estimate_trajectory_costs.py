"""Estimate per-trajectory token counts and USD cost for the `trajectories`
source, which has no cost field of its own (unlike swe_bench_experiments).

Approach:
  - Parse each row's `trajectory` (list of turns: role, text, system_prompt).
  - Treat non-assistant turns (system/user) as input/prompt tokens, and
    assistant turns as output/completion tokens.
  - Token-count with tiktoken's cl100k_base encoding as a proxy. This is NOT
    the actual Llama tokenizer (nebius/SWE-agent-trajectories runs Llama
    8B/70B/405B variants) — cl100k_base typically over/under-counts by
    roughly 10-20% relative to Llama's tokenizer. Good enough for relative
    cost comparison across rows/models; flag this approximation in your
    report/methodology section.
  - Apply per-model $/1M-token pricing (Nebius AI Studio rates as of this
    writing) to get estimated_cost_usd.

Output: overwrites data/raw/trajectories.parquet in place, adding
input_tokens, output_tokens, and estimated_cost_usd columns. Re-run
05_combine_datasets.py afterward to carry these into the combined dataset.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAJECTORIES_PARQUET
from scripts.utils import log, fail, save_df

# $ per 1M tokens, (input, output). Matched against model_name by substring —
# update these if Nebius's published rates have changed since you're reading
# this, or if the dataset's actual model_name strings don't match.
PRICING_PER_1M = {
    "8b": (0.03, 0.03),
    "70b": (0.13, 0.40),
    "405b": (1.00, 3.00),
}


def get_encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def match_pricing(model_name: str):
    if not isinstance(model_name, str):
        return None
    name = model_name.lower()
    for key, price in PRICING_PER_1M.items():
        if key in name:
            return price
    return None


OUTPUT_ROLES = {"assistant", "ai"}  # this dataset labels model turns "ai"


def count_tokens_for_row(trajectory_json, encoder, role_counter=None):
    """Returns (input_tokens, output_tokens) or (None, None) if unparseable."""
    if not isinstance(trajectory_json, str):
        return None, None
    try:
        turns = json.loads(trajectory_json)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(turns, list):
        return None, None

    input_tokens, output_tokens = 0, 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role_counter is not None:
            role_counter[role] = role_counter.get(role, 0) + 1
        text = turn.get("text") or ""
        system_prompt = turn.get("system_prompt") or ""
        combined = f"{system_prompt}{text}"
        n = len(encoder.encode(combined, disallowed_special=()))
        if role in OUTPUT_ROLES:
            output_tokens += n
        else:
            input_tokens += n
    return input_tokens, output_tokens


def main():
    if not os.path.exists(TRAJECTORIES_PARQUET):
        fail(f"{TRAJECTORIES_PARQUET} not found — run 01_load_trajectories.py first")

    df = pd.read_parquet(TRAJECTORIES_PARQUET)
    if "trajectory" not in df.columns:
        fail("no 'trajectory' column found — was this file already stripped by "
             "05_combine_datasets.py? Re-run 01_load_trajectories.py to regenerate "
             "the raw file with full trajectory text before running this script.")

    log(f"estimating tokens/cost for {len(df):,} rows ...")
    encoder = get_encoder()

    input_tok, output_tok = [], []
    role_counter = {}
    for traj in df["trajectory"]:
        i, o = count_tokens_for_row(traj, encoder, role_counter)
        input_tok.append(i)
        output_tok.append(o)

    log(f"turn roles seen across all rows: {role_counter}")
    if not any(r in OUTPUT_ROLES for r in role_counter):
        fail(f"none of the roles {OUTPUT_ROLES} appear in the data (saw {list(role_counter)}) — "
             "add the model-turn role name to OUTPUT_ROLES at the top of this script.")

    df["input_tokens"] = input_tok
    df["output_tokens"] = output_tok

    unmatched_models = set()
    costs = []
    for model_name, in_tok, out_tok in zip(df["model_name"], df["input_tokens"], df["output_tokens"]):
        pricing = match_pricing(model_name)
        if pricing is None:
            unmatched_models.add(model_name)
            costs.append(None)
            continue
        price_in, price_out = pricing
        if in_tok is None or out_tok is None:
            costs.append(None)
            continue
        costs.append((in_tok / 1e6) * price_in + (out_tok / 1e6) * price_out)
    df["estimated_cost_usd"] = costs

    if unmatched_models:
        log(f"WARNING: no pricing match for model_name value(s): {unmatched_models} — "
            f"their estimated_cost_usd is null. Update PRICING_PER_1M in this script.")

    n_priced = df["estimated_cost_usd"].notna().sum()
    log(f"priced {n_priced:,} / {len(df):,} rows "
        f"(mean cost where priced: ${df['estimated_cost_usd'].mean():.4f})")

    save_df(df, TRAJECTORIES_PARQUET)
    log("done — re-run 05_combine_datasets.py to include the new columns")


if __name__ == "__main__":
    main()
