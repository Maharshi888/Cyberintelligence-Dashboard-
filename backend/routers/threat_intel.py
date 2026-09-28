"""
backend/routers/threat_intel.py
---------------------------------
Threat Intelligence Engine API endpoints — unified threat assessment
combining Random Forest, K-Means, and Apriori.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["threat-intelligence"])


class FlowInput(BaseModel):
    """Pydantic model for a single network flow feature dict."""
    features: dict[str, Any]


class BatchFlowInput(BaseModel):
    """Pydantic model for batch flow assessment."""
    flows: list[dict[str, Any]]


# Cache the engine singleton
_engine = None


def _get_engine():
    """Lazily initialise the ThreatIntelligenceEngine singleton."""
    global _engine
    if _engine is None:
        try:
            from src.models.threat_intel import ThreatIntelligenceEngine
            _engine = ThreatIntelligenceEngine()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Threat Intelligence Engine not ready: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialise engine: {e}",
            )
    return _engine


def reset_engine():
    """Reset the engine singleton (e.g. after retraining models)."""
    global _engine
    _engine = None


@router.post("/threat-intel/assess")
def assess_flow(payload: FlowInput):
    """Perform a full threat intelligence assessment on a single network flow.

    Combines:
    - Random Forest classification (prediction + confidence)
    - K-Means cluster assignment (cluster profile + label breakdown)
    - Apriori pattern matching (matched attack patterns)
    - Composite severity scoring

    Returns a unified ThreatAlert with all intelligence combined.
    """
    engine = _get_engine()
    try:
        alert = engine.assess_flow(payload.features)
        return alert.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threat-intel/assess-batch")
def assess_batch(payload: BatchFlowInput):
    """Assess multiple flows at once.

    Returns a list of ThreatAlert dicts.
    """
    engine = _get_engine()
    if len(payload.flows) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 flows.",
        )
    try:
        alerts = engine.assess_batch(payload.flows)
        return {"alerts": [a.to_dict() for a in alerts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threat-intel/summary")
def threat_summary():
    """Return a high-level threat landscape summary.

    Shows which models are active, number of clusters, number of rules,
    top attack patterns, and overall system readiness.
    """
    engine = _get_engine()
    try:
        return engine.get_threat_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threat-intel/reset")
def reset():
    """Reset the engine to reload all model artefacts.

    Call after retraining RF, re-running K-Means, or re-mining Apriori rules.
    """
    reset_engine()
    return {"status": "ok", "message": "Engine reset. Will reload on next request."}
