"""
src/data/preprocessing.py
--------------------------
Data cleaning, encoding, and scaling utilities for the CICIDS2017 dataset.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
LABEL_COL = "Label"


def clean_dataframe(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Replace inf values and fill NaN with column medians.

    Args:
        df: Input DataFrame.
        feature_cols: Columns to process; defaults to all numeric columns.

    Returns:
        Cleaned DataFrame.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    return df


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Fit a LabelEncoder and transform string labels to integers.

    Returns:
        Tuple of (encoded_array, fitted_encoder).
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str).str.strip())
    return encoded, encoder


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """Fit a StandardScaler on training data and transform train+test.

    Args:
        X_train: Training features DataFrame.
        X_test: Optional test features DataFrame.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled_or_None, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def load_scaler() -> StandardScaler:
    """Load the saved StandardScaler from model artifacts."""
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_label_encoder() -> LabelEncoder:
    """Load the saved LabelEncoder from model artifacts."""
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)


def prepare_inference_row(
    row: dict,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """Prepare a single input row dict for model inference.

    Missing features are filled with 0. Returns a scaled 2-D array.
    """
    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[feature_columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scaler.transform(df)
