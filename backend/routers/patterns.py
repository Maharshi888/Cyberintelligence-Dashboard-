"""
backend/routers/patterns.py
-----------------------------
Apriori association rule mining endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/api", tags=["patterns"])

_mining_status: dict = {"status": "idle", "message": ""}


@router.get("/patterns")
def get_patterns(top_n: int = 20):
    """Return cached Apriori association rules sorted by lift.

    Returns status='not_computed' if mining hasn't been triggered yet.
    """
    from src.models.apriori import get_top_rules

    rules = get_top_rules(n=top_n)
    if not rules:
        return {"status": "not_computed", "rules": []}
    return {"status": "ready", "rules": rules}


@router.post("/patterns/mine")
def trigger_mining(
    background_tasks: BackgroundTasks,
    sample_size: int = 30000,
    min_support: float = 0.05,
    min_confidence: float = 0.6,
):
    """Trigger Apriori mining in the background.

    Args:
        sample_size: Rows to sample (default 30 000).
        min_support: Support threshold (0-1).
        min_confidence: Confidence threshold (0-1).
    """
    global _mining_status
    if _mining_status["status"] == "running":
        return {"status": "running", "message": "Mining already in progress."}

    _mining_status = {"status": "running", "message": "Mining started…"}
    background_tasks.add_task(
        _run_mining,
        sample_size=sample_size,
        min_support=min_support,
        min_confidence=min_confidence,
    )
    return {
        "status": "running",
        "message": f"Apriori mining started (sample={sample_size}, support≥{min_support}, confidence≥{min_confidence}).",
    }


@router.get("/patterns/status")
def mining_status():
    """Return the current Apriori mining job status."""
    return _mining_status


def _run_mining(sample_size: int, min_support: float, min_confidence: float) -> None:
    """Background task: sample data, build transactions, mine rules."""
    global _mining_status
    try:
        from src.data.loader import get_stratified_sample
        from src.models.apriori import mine_patterns

        _mining_status["message"] = "Loading data sample…"
        df = get_stratified_sample(n=sample_size)

        _mining_status["message"] = "Building transaction matrix…"
        rules = mine_patterns(df, min_support=min_support, min_confidence=min_confidence)

        _mining_status = {
            "status": "done",
            "message": f"Completed. {len(rules)} rules discovered.",
        }
    except Exception as exc:
        _mining_status = {"status": "error", "message": str(exc)}
