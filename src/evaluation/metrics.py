"""
src/evaluation/metrics.py
--------------------------
Evaluation helpers for classification, clustering, and model reporting.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Compute per-class precision, recall, F1 and overall accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        classes: List of class name strings indexed by integer label.

    Returns:
        Dict with 'accuracy', 'macro_f1', and per-class 'per_class' metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report

    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in classes
            if cls in report
        },
    }


def cluster_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean silhouette coefficient for a clustering result.

    Returns -1.0 if fewer than 2 clusters are present (degenerate case).
    """
    from sklearn.metrics import silhouette_score

    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(X):
        return -1.0
    # Subsample for speed if dataset is large
    if len(X) > 10_000:
        idx = np.random.choice(len(X), 10_000, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def confusion_matrix_b64(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    """Generate a confusion matrix heatmap and return it as a base64-encoded PNG.

    Useful for embedding directly in API responses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_model_report() -> dict:
    """Load the cached model evaluation report from reports/model_report.json."""
    path = REPORTS_DIR / "model_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
