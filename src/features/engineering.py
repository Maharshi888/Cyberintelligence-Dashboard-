"""
src/features/engineering.py
----------------------------
Feature engineering utilities: temporal aggregation, flag ratios,
and feature importance extraction for the CICIDS2017 dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
LABEL_COL = "Label"

# TCP flag columns present in CICIDS2017
FLAG_COLS = [
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
]


def compute_flag_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add flag-ratio columns (each flag count / total packets).

    Returns the DataFrame with new columns appended (does not modify in place).
    """
    df = df.copy()
    total = df.get("Total Fwd Packets", pd.Series(1, index=df.index)) + df.get(
        "Total Backward Packets", pd.Series(1, index=df.index)
    )
    total = total.replace(0, 1)  # avoid divide-by-zero
    for flag in FLAG_COLS:
        if flag in df.columns:
            df[f"{flag} Ratio"] = df[flag] / total
    return df


def compute_packet_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived packet-size statistics."""
    df = df.copy()
    if "Total Length of Fwd Packets" in df.columns and "Total Fwd Packets" in df.columns:
        df["Avg Fwd Payload"] = df["Total Length of Fwd Packets"] / (
            df["Total Fwd Packets"].replace(0, 1)
        )
    if "Total Length of Bwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        df["Avg Bwd Payload"] = df["Total Length of Bwd Packets"] / (
            df["Total Backward Packets"].replace(0, 1)
        )
    return df


def get_top_features(n: int = 20) -> list[dict]:
    """Load the top-N features by Random Forest feature importance.

    Returns a list of dicts: [{"feature": str, "importance": float}, ...]
    Reads from the saved model artifacts.
    """
    importance_path = MODEL_DIR / "feature_importances.json"
    if importance_path.exists():
        with open(importance_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:n]

    # Fallback: compute from model if available
    model_path = MODEL_DIR / "random_forest.pkl"
    features_path = MODEL_DIR / "feature_columns.json"
    if not model_path.exists() or not features_path.exists():
        return []

    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    result = [
        {"feature": feature_cols[i], "importance": float(importances[i])}
        for i in idx
    ]
    # Cache for next call
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result[:n]
