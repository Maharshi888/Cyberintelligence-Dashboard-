from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def load_model_bundle():
    with open(MODEL_DIR / "random_forest.pkl", "rb") as file:
        model = pickle.load(file)

    with open(MODEL_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open(MODEL_DIR / "label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    with open(MODEL_DIR / "feature_columns.json", "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    return model, scaler, encoder, feature_columns


def preprocess_input(row: dict) -> pd.DataFrame:
    model, scaler, encoder, feature_columns = load_model_bundle()
    # Input row can contain any column subset; missing values get defaults.
    raw = pd.DataFrame([row])
    raw = raw[feature_columns] if all(col in raw.columns for col in feature_columns) else raw

    # Coerce numerics and impute missing values with training medians.
    for col in feature_columns:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw[feature_columns]
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(raw.median(numeric_only=True))

    scaled = scaler.transform(raw)
    return model.predict_proba(scaled), model.predict(scaled), encoder


def predict(row: dict):
    probabilities, prediction, encoder = preprocess_input(row)
    class_index = int(prediction[0])
    predicted_label = encoder.inverse_transform([class_index])[0]
    confidence = float(np.max(probabilities[0]))
    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, probabilities[0])},
    }


if __name__ == "__main__":
    sample = {
        "Destination Port": 80,
        "Flow Duration": 100,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 1,
        "Total Length of Fwd Packets": 12,
        "Total Length of Bwd Packets": 12,
        "Label": "BENIGN",
    }
    print(predict(sample))
