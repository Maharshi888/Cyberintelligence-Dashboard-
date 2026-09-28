"""
backend/routers/analytics.py
------------------------------
Dataset-level analytics endpoints: label distribution, attack trends,
model report, and feature importances.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

router = APIRouter(prefix="/api", tags=["analytics"])


def _read_json(path: Path) -> dict | list:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data not found: {path.name}. Run the training pipeline first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/stats")
def get_stats():
    """Return label distribution and basic dataset statistics."""
    dist = _read_json(PROCESSED_DIR / "label_distribution.json")
    total = sum(dist.values())
    attack_count = sum(v for k, v in dist.items() if k.upper() != "BENIGN")
    return {
        "total_flows": total,
        "attack_flows": attack_count,
        "benign_flows": total - attack_count,
        "attack_types": len([k for k in dist if k.upper() != "BENIGN"]),
        "label_distribution": dist,
    }


@router.get("/trends")
def get_trends():
    """Return simulated hourly attack trend data for the chart."""
    trends_path = PROCESSED_DIR / "attack_trends.json"
    return _read_json(trends_path)


@router.get("/model-report")
def get_model_report():
    """Return the Random Forest evaluation report."""
    return _read_json(REPORTS_DIR / "model_report.json")


@router.get("/features")
def get_feature_importances(top_n: int = 20):
    """Return top-N feature importances from the Random Forest model."""
    from src.models.random_forest import get_feature_importances
    try:
        features = get_feature_importances(top_n=top_n)
        if not features:
            raise HTTPException(status_code=404, detail="Feature importances not available. Train the model first.")
        return {"features": features}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/forensics")
def get_forensics():
    """Return authentic dataset forensics: top targeted ports and TCP flag radar profile."""
    forensics_path = PROCESSED_DIR / "forensics.json"
    if not forensics_path.exists():
        # Fallback to computing or basic distribution if not generated yet
        raise HTTPException(
            status_code=404,
            detail="Forensics dataset metrics not yet computed. Run compute_forensics.py.",
        )
    return _read_json(forensics_path)

