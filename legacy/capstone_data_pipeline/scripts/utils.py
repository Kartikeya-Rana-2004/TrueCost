"""Shared helpers for the pipeline scripts."""
import json
import sys

import pandas as pd


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[pipeline][ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _is_nested(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, dict, tuple)):
        return True
    # numpy arrays, pandas/pyarrow array-likes, etc. — anything array-like
    # that isn't a string/bytes scalar
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return True
    return False


def _to_json_safe(v):
    if v is None:
        return None
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes, list, dict, tuple)):
        v = v.tolist()
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=str)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _sanitize_for_parquet(df):
    """JSON-stringify any object column containing nested/array-like values
    (lists, dicts, tuples, numpy arrays, ...). Parquet requires one type per
    column, and pyarrow can silently write a mixed nested column only to
    fail with 'Nested data conversions not implemented' on read — so we
    flatten anything nested to a JSON string up front.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        try:
            if not df[col].map(_is_nested).any():
                continue
            df[col] = df[col].map(_to_json_safe)
        except Exception as e:
            log(f"WARNING: could not sanitize column '{col}' ({e}) — leaving as-is")
    return df


def save_df(df, path: str) -> None:
    df = _sanitize_for_parquet(df)
    df.to_parquet(path, index=False)
    log(f"wrote {len(df):,} rows -> {path}")


def normalize_instance_id_col(df, candidates):
    """Rename the first matching column in `candidates` to `instance_id`."""
    for c in candidates:
        if c in df.columns:
            if c != "instance_id":
                df = df.rename(columns={c: "instance_id"})
            return df
    return df
