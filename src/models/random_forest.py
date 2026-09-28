"""
src/models/random_forest.py
----------------------------
Random Forest threat classifier — wraps the training pipeline in
train_random_forest.py and exposes prediction + feature-importance APIs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def is_trained() -> bool:
    """Return True if all required model artefacts are present on disk."""
    required = [
        MODEL_DIR / "random_forest.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "label_encoder.pkl",
        MODEL_DIR / "feature_columns.json",
    ]
    return all(p.exists() for p in required)


def load_bundle() -> tuple:
    """Load (model, scaler, encoder, feature_columns) from disk.

    Raises FileNotFoundError if artefacts are missing.
    """
    if not is_trained():
        raise FileNotFoundError(
            "Model artefacts not found. Run `python src/train_random_forest.py` first."
        )
    with open(MODEL_DIR / "random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model, scaler, encoder, feature_columns


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_single(row: dict, bundle: Optional[tuple] = None) -> dict:
    """Classify a single network flow described by a feature dict.

    Args:
        row: Dict mapping feature names to numeric values. Missing keys → 0.
        bundle: Optional pre-loaded (model, scaler, encoder, feature_columns).

    Returns:
        {"prediction": str, "confidence": float, "probabilities": {label: float}}
    """
    if bundle is None:
        bundle = load_bundle()
    model, scaler, encoder, feature_columns = bundle

    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaled = scaler.transform(df)
    pred_idx = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]

    return {
        "prediction": encoder.inverse_transform([pred_idx])[0],
        "confidence": float(np.max(proba)),
        "probabilities": {
            cls: float(p) for cls, p in zip(encoder.classes_, proba)
        },
    }


def predict_batch(rows: list[dict], bundle: Optional[tuple] = None) -> list[dict]:
    """Classify a batch of flow dicts. Returns a list of prediction dicts."""
    if bundle is None:
        bundle = load_bundle()
    return [predict_single(row, bundle) for row in rows]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importances(top_n: int = 20, bundle: Optional[tuple] = None) -> list[dict]:
    """Return the top-N features ranked by Random Forest importance.

    Caches to model/feature_importances.json.

    Returns:
        [{"feature": str, "importance": float}, ...]
    """
    cache_path = MODEL_DIR / "feature_importances.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:top_n]

    if bundle is None:
        bundle = load_bundle()
    model, _, _, feature_columns = bundle

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    result = [
        {"feature": feature_columns[i], "importance": float(importances[i])}
        for i in idx
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result[:top_n]
