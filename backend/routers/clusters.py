"""
backend/routers/clusters.py
-----------------------------
K-Means clustering endpoints: trigger training and retrieve cluster profiles.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/api", tags=["clustering"])

_training_status: dict = {"status": "idle", "message": ""}


@router.get("/clusters")
def get_clusters():
    """Return cached K-Means cluster profiles.

    If clustering hasn't been run yet, returns status='not_computed' so the
    frontend can show a 'Run Clustering' button.
    """
    from src.models.kmeans import get_cluster_profiles

    profiles = get_cluster_profiles()
    if not profiles:
        return {"status": "not_computed", "clusters": []}
    return {"status": "ready", **profiles}


@router.post("/clusters/train")
def trigger_clustering(background_tasks: BackgroundTasks, k: int = 8, sample_size: int = 50000):
    """Trigger K-Means training in the background.

    Args:
        k: Number of clusters (default 8).
        sample_size: Number of rows to sample from the full dataset.
    """
    global _training_status
    if _training_status["status"] == "running":
        return {"status": "running", "message": "Clustering already in progress."}

    _training_status = {"status": "running", "message": "Clustering started…"}
    background_tasks.add_task(_run_clustering, k=k, sample_size=sample_size)
    return {"status": "running", "message": f"K-Means training started with k={k}, sample={sample_size} rows."}


@router.get("/clusters/status")
def clustering_status():
    """Return the current clustering job status."""
    return _training_status


def _run_clustering(k: int, sample_size: int) -> None:
    """Background task: load data, train K-Means, save profiles."""
    global _training_status
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        from src.data.loader import get_stratified_sample, get_numeric_columns
        from src.data.preprocessing import clean_dataframe
        from src.models.kmeans import train_kmeans

        _training_status["message"] = "Loading data sample…"
        df = get_stratified_sample(n=sample_size)

        _training_status["message"] = "Preprocessing features…"
        numeric_cols = get_numeric_columns(df)
        df_clean = clean_dataframe(df, feature_cols=numeric_cols)
        X = df_clean[numeric_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        labels = df["Label"] if "Label" in df.columns else None

        _training_status["message"] = f"Running KMeans (k={k})…"
        result = train_kmeans(
            X_scaled,
            feature_names=numeric_cols,
            labels=labels,
            original_df=df_clean,
            k=k,
        )

        _training_status = {
            "status": "done",
            "message": f"Completed. Silhouette score: {result['silhouette_score']:.4f}",
        }
    except Exception as exc:
        _training_status = {"status": "error", "message": str(exc)}
