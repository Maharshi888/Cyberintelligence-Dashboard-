from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "raw" / "combinenew.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

LABEL_COL = "Label"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([float("inf"), float("-inf")], np.nan)
    return df


def build_model_artifacts():
    df = load_dataset(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Could not find required label column: {LABEL_COL}")

    y = df[LABEL_COL].astype(str).str.strip()
    drop_columns = [LABEL_COL]
    X = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    labels = y.values.astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train_scaled, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "random_forest.pkl", "wb") as file:
        pickle.dump(rf, file)

    with open(MODEL_DIR / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)

    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(numeric_cols, file, indent=2)

    y_pred = rf.predict(X_test_scaled)
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    with open(REPORT_DIR / "model_report.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "label_count": len(encoder.classes_),
                "classes": encoder.classes_.tolist(),
                "classification_report": report,
            },
            file,
            indent=2,
            default=str,
        )

    print(f"Saved model artifacts to {MODEL_DIR}")
    print(f"Dataset shape: {df.shape}")
    print(f"Features used: {len(numeric_cols)}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    build_model_artifacts()
