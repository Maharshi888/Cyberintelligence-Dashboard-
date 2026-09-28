"""
src/data/loader.py
------------------
Robust utilities for loading the CICIDS2017 dataset efficiently.
Provides memory-safe chunked reading, stratified sampling, and cached flow pooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "raw" / "combinenew.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"


def load_raw(path: Path = RAW_CSV, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw CICIDS2017 CSV file.

    Args:
        path: Path to the CSV file.
        nrows: If set, load only the first N rows.

    Returns:
        DataFrame with stripped column names and inf values replaced with NaN.
    """
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_label_distribution(df: Optional[pd.DataFrame] = None) -> dict[str, int]:
    """Return the count of each attack label.

    Caches result to processed/label_distribution.json for fast re-use.
    """
    cache = PROCESSED_DIR / "label_distribution.json"
    if cache.exists() and df is None:
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)

    if df is not None:
        dist = df[LABEL_COL].astype(str).str.strip().value_counts().to_dict()
    else:
        # Chunked label distribution calculation (RAM safe)
        dist = {}
        for chunk in pd.read_csv(RAW_CSV, chunksize=250_000, low_memory=False, usecols=[LABEL_COL]):
            for lbl, cnt in chunk[LABEL_COL].astype(str).str.strip().value_counts().items():
                dist[lbl] = dist.get(lbl, 0) + int(cnt)

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2)
    return dist


def get_stratified_sample(
    n: int = 50_000, df: Optional[pd.DataFrame] = None, random_state: int = 42
) -> pd.DataFrame:
    """Return a stratified random sample of N rows from the 2.83M dataset using safe chunking."""
    if df is not None:
        label_col = LABEL_COL if LABEL_COL in df.columns else df.columns[-1]
        frac = min(n / max(1, len(df)), 1.0)
        return (
            df.groupby(label_col, group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=random_state))
            .reset_index(drop=True)
        )

    # Chunked stratified sampling from disk without loading 872MB at once
    samples = []
    chunksize = 250_000
    rows_per_chunk = max(100, int(n / 12))  # ~12 chunks in 2.83M dataset

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=chunksize, low_memory=False)):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        label_col = LABEL_COL if LABEL_COL in chunk.columns else chunk.columns[-1]
        
        take = min(rows_per_chunk, len(chunk))
        sampled_chunk = (
            chunk.groupby(label_col, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, int(len(g) / len(chunk) * take))), random_state=random_state + i))
            .reset_index(drop=True)
        )
        samples.append(sampled_chunk)

    combined = pd.concat(samples, ignore_index=True)
    if len(combined) > n:
        combined = combined.sample(n=n, random_state=random_state).reset_index(drop=True)
    return combined


def get_cached_sample_pool(pool_size: int = 3000) -> pd.DataFrame:
    """Return an authentic pool of dataset rows cached locally for fast sub-millisecond inference."""
    cache_path = PROCESSED_DIR / "sample_pool.pkl"
    if cache_path.exists():
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            pass

    df = get_stratified_sample(n=pool_size)
    try:
        df.to_pickle(cache_path)
    except Exception:
        pass
    return df


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return names of all numeric columns excluding the label."""
    return [
        c
        for c in df.select_dtypes(include=np.number).columns.tolist()
        if c != LABEL_COL
    ]
