"""
src/models/random_forest.py
----------------------------
Random Forest threat classifier — wraps the training pipeline in
train_random_forest.py and exposes prediction + feature-importance APIs.

Fix #1  — predict_single reports missing_features and feature_coverage so the
          caller (and UI) knows inputs are incomplete rather than silently using
          zero-filled phantom values.
Fix #9  — Saved imputer_medians.json (written by train_random_forest.py) is used
          at inference; falls back to 0.0 only when the file is absent (legacy).
Fix #20 — feature_coverage, supplied_features, total_features and missing_features
          are included in every predict_single response.
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
    """Load (model, scaler, encoder, feature_columns, imputer_medians) from disk.

    imputer_medians is a dict {feature: median} written by train_random_forest.py.
    If the JSON file doesn't exist (legacy artefacts) an empty dict is used and
    missing features fall back to 0.0.

    Raises FileNotFoundError if core artefacts are missing.
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

    # Fix #9 — load training-set medians for consistent imputation
    imputer_medians: dict[str, float] = {}
    medians_path = MODEL_DIR / "imputer_medians.json"
    if medians_path.exists():
        with open(medians_path, "r", encoding="utf-8") as f:
            imputer_medians = json.load(f)

    return model, scaler, encoder, feature_columns, imputer_medians


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _is_blank(v) -> bool:
    """Return True if a value should be treated as missing/invalid."""
    if v is None:
        return True
    try:
        f = float(v)
        return not np.isfinite(f)
    except (TypeError, ValueError):
        return True


def predict_single(row: dict, bundle: Optional[tuple] = None) -> dict:
    """Classify a single network flow described by a feature dict.

    Args:
        row: Dict mapping feature names to numeric values.
             Missing/NaN/inf keys are filled with training-set medians (Fix #9).
        bundle: Optional pre-loaded bundle from load_bundle().

    Returns:
        {
          "prediction": str,
          "confidence": float,           # max class probability
          "probabilities": {label: float},
          "feature_coverage": float,     # fraction of 77 features provided (Fix #1)
          "supplied_features": int,
          "total_features": int,
          "missing_features": [str],     # up to 20 feature names (Fix #20)
        }
    """
    if bundle is None:
        bundle = load_bundle()

    # Handle legacy 4-tuple bundles from old code paths
    if len(bundle) == 4:
        model, scaler, encoder, feature_columns = bundle
        imputer_medians: dict[str, float] = {}
    else:
        model, scaler, encoder, feature_columns, imputer_medians = bundle

    # Fix #1 — track which model features were genuinely supplied
    supplied = {k for k in row if k in feature_columns and not _is_blank(row.get(k))}
    missing = [c for c in feature_columns if c not in supplied]
    coverage = len(supplied) / max(len(feature_columns), 1)

    # Fix #9 — build feature vector using training medians for missing/invalid values
    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = imputer_medians.get(col, 0.0)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[feature_columns]

    # Replace inf and NaN with training medians (or 0.0 as last resort)
    for col in feature_columns:
        bad_mask = ~np.isfinite(df[col].values.astype(float))
        if bad_mask.any():
            df.loc[df.index[bad_mask], col] = imputer_medians.get(col, 0.0)

    scaled = scaler.transform(df)
    pred_idx = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]

    return {
        "prediction": encoder.inverse_transform([pred_idx])[0],
        "confidence": float(np.max(proba)),
        "probabilities": {
            cls: float(p) for cls, p in zip(encoder.classes_, proba)
        },
        # Fix #1 / #20 — coverage metadata
        "feature_coverage": round(coverage, 4),
        "supplied_features": len(supplied),
        "total_features": len(feature_columns),
        "missing_features": missing[:20],   # cap list length for JSON response
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

    if len(bundle) == 4:
        model, _, _, feature_columns = bundle
    else:
        model, _, _, feature_columns, _ = bundle

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    result = [
        {"feature": feature_columns[i], "importance": float(importances[i])}
        for i in idx
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result[:top_n]
