"""
backend/app.py
---------------
FastAPI application entry point for the Cyber Threat Intelligence Dashboard.

Startup behaviour:
  1. Pre-loads the RF model bundle (if trained) and stores in app.state.
  2. Pre-computes label distribution and attack trend cache.
  3. Serves the frontend SPA as static files from ../frontend/.

Run with:
    python backend/app.py
  or:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import analytics, clusters, patterns, predict, threat_intel

FRONTEND_DIR = ROOT / "frontend"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lifespan: warm up model and pre-compute analytics cache on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — runs on startup, cleans up on shutdown."""
    # ── Startup ──────────────────────────────────────────────────────────────
    print("[startup] Warming up model bundle…")
    try:
        from src.models.random_forest import load_bundle, is_trained, get_feature_importances

        if is_trained():
            bundle = load_bundle()
            app.state.rf_bundle = bundle
            get_feature_importances(top_n=20, bundle=bundle)
            print("[startup] RF model loaded successfully.")
        else:
            app.state.rf_bundle = None
            print("[startup] RF model not found — run src/train_random_forest.py to train.")
    except Exception as exc:
        app.state.rf_bundle = None
        print(f"[startup] Model load warning: {exc}")

    print("[startup] Building analytics cache…")
    try:
        from src.data.loader import get_label_distribution

        dist = get_label_distribution()
        _build_trend_cache(dist)
        print(f"[startup] Label distribution cached ({len(dist)} classes).")
    except Exception as exc:
        print(f"[startup] Analytics cache warning: {exc}")

    print("[startup] Ready.")

    yield  # ── Application runs here ──

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[shutdown] Cyber Threat Intelligence API shutting down.")


app = FastAPI(
    title="Cyber Threat Intelligence Dashboard API",
    description=(
        "REST API for RF classification, K-Means clustering, "
        "Apriori pattern mining, and unified threat intelligence on CICIDS2017."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow browser requests from any origin (development convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analytics.router)
app.include_router(predict.router)
app.include_router(clusters.router)
app.include_router(patterns.router)
app.include_router(threat_intel.router)


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health_check():
    """System health check and loaded services status."""
    from src.models.random_forest import is_trained
    return {
        "status": "healthy",
        "service": "Cyber Threat Intelligence API",
        "model_trained": is_trained(),
        "version": "2.0.0",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trend_cache(dist: dict) -> None:
    """Generate a synthetic hourly attack-trend cache from label distribution.

    In a production system this would be derived from timestamps in the data.
    Since CICIDS2017 lacks absolute timestamps, we simulate realistic trends
    that match the true label proportions.
    """
    import math
    import random

    trend_path = PROCESSED_DIR / "attack_trends.json"
    if trend_path.exists():
        return  # Already cached

    random.seed(42)
    labels = [k for k in dist if k.upper() != "BENIGN"]
    total = sum(dist.values())

    hours = list(range(24))
    series: dict[str, list[int]] = {}

    for label in labels:
        proportion = dist.get(label, 0) / total
        base = max(1, int(proportion * 5000))
        values = []
        for h in hours:
            # Simulate diurnal attack pattern: peaks at business hours
            diurnal = 1 + 0.6 * math.sin(math.pi * (h - 6) / 12)
            noise = random.uniform(0.8, 1.2)
            values.append(max(0, int(base * diurnal * noise)))
        series[label] = values

    result = {
        "hours": [f"{h:02d}:00" for h in hours],
        "series": series,
    }
    with open(trend_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Static file serving — frontend SPA
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    # Serve static assets (CSS, JS)
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
