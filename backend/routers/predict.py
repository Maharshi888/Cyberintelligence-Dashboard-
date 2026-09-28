"""
backend/routers/predict.py
---------------------------
Single-flow prediction endpoint and genuine dataset inference streaming using
the trained Random Forest model bundle.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any
import pandas as pd
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["prediction"])

# In-memory rolling feed — stores the last 50 genuine predictions
_live_feed: deque[dict] = deque(maxlen=50)

SEVERITY_MAP = {
    "BENIGN": "low",
    "DoS Hulk": "critical",
    "PortScan": "high",
    "DDoS": "critical",
    "DoS GoldenEye": "high",
    "FTP-Patator": "medium",
    "SSH-Patator": "medium",
    "DoS slowloris": "high",
    "DoS Slowhttptest": "high",
    "Bot": "critical",
    "Web Attack – Brute Force": "high",
    "Web Attack_Brute Force": "high",
    "Web Attack – XSS": "medium",
    "Web Attack_XSS": "medium",
    "Infiltration": "critical",
    "Web Attack – Sql Injection": "high",
    "Web Attack_Sql Injection": "high",
    "Heartbleed": "critical",
}


def _severity(label: str) -> str:
    return SEVERITY_MAP.get(label, "medium" if label.upper() != "BENIGN" else "low")


class FlowInput(BaseModel):
    """Pydantic model for a single network flow feature dict."""
    features: dict[str, Any]


@router.post("/predict")
def predict_flow(payload: FlowInput):
    """Classify a single network flow.

    Accepts a JSON body: {"features": {"Destination Port": 80, ...}}
    Returns label, confidence, and per-class probabilities.
    """
    from src.models.random_forest import load_bundle, predict_single, is_trained
    if not is_trained():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python src/train_random_forest.py` first.",
        )
    try:
        t0 = time.perf_counter()
        bundle = load_bundle()
        result = predict_single(payload.features, bundle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prediction": result["prediction"],
            "true_label": payload.features.get("True_Label", "Injected Flow"),
            "confidence": result["confidence"],
            "severity": _severity(result["prediction"]),
            "latency_ms": round(elapsed_ms, 2),
            "source": "User / Telemetry Injection",
        }
        _live_feed.appendleft(entry)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live-feed")
def get_live_feed(limit: int = 20):
    """Return the most recent genuine threat predictions as a live feed.

    If no manual predictions have been submitted yet, populates feed by
    evaluating genuine dataset rows against the trained Random Forest model.
    """
    from src.models.random_forest import is_trained, load_bundle, predict_single
    from src.data.loader import get_cached_sample_pool

    feed = list(_live_feed)[:limit]

    # If empty, run real inference on a batch of genuine dataset flows
    if not feed and is_trained():
        try:
            bundle = load_bundle()
            sample_df = get_cached_sample_pool(pool_size=5000).sample(n=min(limit, 15), random_state=42)
            for _, row in sample_df.iterrows():
                true_label = str(row["Label"]).strip() if "Label" in row else "UNKNOWN"
                features = row.to_dict()
                
                t0 = time.perf_counter()
                res = predict_single(features, bundle)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "prediction": res["prediction"],
                    "true_label": true_label,
                    "confidence": res["confidence"],
                    "severity": _severity(res["prediction"]),
                    "latency_ms": round(elapsed_ms, 2),
                    "source": "Dataset Validation Flow",
                }
                _live_feed.appendleft(entry)
            feed = list(_live_feed)[:limit]
        except Exception as err:
            print(f"[live-feed] Warning evaluating dataset flows: {err}")

    return {"feed": feed}


@router.post("/stream-dataset-flow")
def stream_dataset_flow(n: int = 5):
    """Run real inference on N random authentic flows from the dataset and append to feed."""
    from src.models.random_forest import is_trained, load_bundle, predict_single
    from src.data.loader import get_cached_sample_pool

    if not is_trained():
        raise HTTPException(status_code=503, detail="Model not trained.")

    bundle = load_bundle()
    pool = get_cached_sample_pool(pool_size=5000)
    sample_df = pool.sample(n=min(n, len(pool)))
    evaluated = []

    for _, row in sample_df.iterrows():
        true_label = str(row["Label"]).strip() if "Label" in row else "UNKNOWN"
        features = row.to_dict()
        
        t0 = time.perf_counter()
        res = predict_single(features, bundle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prediction": res["prediction"],
            "true_label": true_label,
            "confidence": res["confidence"],
            "severity": _severity(res["prediction"]),
            "latency_ms": round(elapsed_ms, 2),
            "source": "Dataset Validation Flow",
        }
        _live_feed.appendleft(entry)
        evaluated.append(entry)

    return {"status": "ok", "new_flows": evaluated}

