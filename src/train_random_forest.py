from pathlib import Path
import json
import pickle
import hashlib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

LABEL_COL = "Label"


def load_dataset(path=None, nrows=None):
    """Load dataset using src.data.loader.load_raw."""
    from src.data.loader import load_raw, RAW_CSV
    csv_path = path if path is not None else RAW_CSV
    return load_raw(csv_path, nrows=nrows)


def build_model_artifacts(sample_size=None, n_estimators=100):
    """Train the Random Forest and save all artefacts.

    Fix #10: deduplication before split; imputation computed on train fold ONLY.
    Fix #9:  saves model/imputer_medians.json so inference preprocessing matches.
    Fix #11: saves model/manifest.json (hash, feature list, sklearn version, seed).
    """
    import sklearn

    df = load_dataset()

    if LABEL_COL not in df.columns:
        raise ValueError("Could not find required label column: " + LABEL_COL)

    # Fix #10a — deduplicate (CIC-IDS2017 contains ~10% near-duplicate rows)
    before = len(df)
    df = df.drop_duplicates()
    n_removed = before - len(df)
    print("Deduplication: {} -> {} rows ({} dupes removed).".format(before, len(df), n_removed))

    if sample_size and len(df) > sample_size:
        print("Sampling {} flows from {} total records...".format(sample_size, len(df)))
        try:
            df = df.groupby(LABEL_COL, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(2, int(len(x) / len(df) * sample_size))),
                                   random_state=42)
            ).reset_index(drop=True)
        except Exception:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    y = df[LABEL_COL].astype(str).str.strip()
    X = df.drop(columns=[LABEL_COL], errors="ignore")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    labels = y.values.astype(str)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)

    # Fix #10b — split BEFORE imputation to prevent test-set information leaking
    # into the imputation medians used to fill training data.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_encoded, np.arange(len(X)),
        test_size=0.2, random_state=42, stratify=y_encoded,
    )

    # Fix #9 — compute medians on training fold only, then use for both folds
    train_medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(pd.Series(train_medians))
    X_test = X_test.fillna(pd.Series(train_medians))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest ({} estimators) on {} samples...".format(
        n_estimators, X_train_scaled.shape[0]))
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train_scaled, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(numeric_cols, f, indent=2)

    # Fix #9 — persist imputer medians for identical serve-time preprocessing
    safe_medians = {}
    for k, v in train_medians.items():
        try:
            fv = float(v)
            safe_medians[k] = fv if np.isfinite(fv) else 0.0
        except (TypeError, ValueError):
            safe_medians[k] = 0.0
    with open(MODEL_DIR / "imputer_medians.json", "w", encoding="utf-8") as f:
        json.dump(safe_medians, f, indent=2)

    # Fix #11 — save split indices so the exact held-out set is reusable
    np.save(MODEL_DIR / "test_idx.npy", idx_test)
    np.save(MODEL_DIR / "train_idx.npy", idx_train)

    y_pred = rf.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0, output_dict=True,
    )
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    acc = accuracy_score(y_test, y_pred)
    print("  Accuracy : {:.4f}".format(acc))
    print("  Macro-F1 : {:.4f}  <- the informative metric (not accuracy alone)".format(macro_f1))

    full_report = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "label_count": len(encoder.classes_),
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
    }
    with open(REPORT_DIR / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)

    # Fix #11 — provenance manifest for reproducibility auditing
    with open(MODEL_DIR / "random_forest.pkl", "rb") as f:
        pkl_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    manifest = {
        "sklearn_version": sklearn.__version__,
        "n_estimators": n_estimators,
        "n_features": len(numeric_cols),
        "feature_columns": numeric_cols,
        "n_classes": len(encoder.classes_),
        "classes": encoder.classes_.tolist(),
        "random_state": 42,
        "test_size": 0.2,
        "sample_size": sample_size,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dedup": True,
        "imputer": "median (train fold only — saved to imputer_medians.json)",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "pkl_sha256_prefix": pkl_hash,
    }
    with open(MODEL_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved model artifacts to {}".format(MODEL_DIR))
    print("Dataset shape: {}  |  Features: {}  |  Macro-F1: {:.4f}".format(
        df.shape, len(numeric_cols), macro_f1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Random Forest Classifier on CICIDS2017")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (e.g. 50000)")
    parser.add_argument("--estimators", type=int, default=100, help="Number of trees (default: 100)")
    args = parser.parse_args()
    build_model_artifacts(sample_size=args.sample, n_estimators=args.estimators)