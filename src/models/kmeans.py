"""
src/models/kmeans.py
---------------------
K-Means attack clustering for the CICIDS2017 dataset.

Usage:
    from src.models.kmeans import train_kmeans, get_cluster_profiles, load_kmeans

Workflow:
    1. Call train_kmeans() to fit and persist the model.
    2. Call get_cluster_profiles() to retrieve cluster summaries for the dashboard.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

KMEANS_PATH = MODEL_DIR / "kmeans.pkl"
PROFILES_PATH = PROCESSED_DIR / "cluster_profiles.json"


def train_kmeans(
    X: np.ndarray,
    feature_names: list[str],
    labels: Optional[pd.Series] = None,
    original_df: Optional[pd.DataFrame] = None,
    k: int = 8,
    random_state: int = 42,
) -> dict:
    """Fit KMeans on feature matrix X and persist the model.

    Args:
        X: Scaled feature matrix (n_samples, n_features).
        feature_names: Column names corresponding to X's columns.
        labels: Optional original string labels for each sample (used to
                build a dominant-attack mapping per cluster).
        original_df: Optional original DataFrame with raw (unscaled) values
                     for computing rich network profiles per cluster.
        k: Number of clusters.
        random_state: Random seed.

    Returns:
        Cluster profiles dict (also saved to processed/cluster_profiles.json).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = km.fit_predict(X)

    # Silhouette score (subsample for speed)
    sample_size = min(10_000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    sil = float(silhouette_score(X[idx], cluster_labels[idx]))

    # Build cluster profiles
    profiles = []
    for cid in range(k):
        mask = cluster_labels == cid
        count = int(mask.sum())
        centroid = km.cluster_centers_[cid].tolist()

        # Dominant attack type + full label breakdown
        dominant = "Unknown"
        label_breakdown: dict[str, float] = {}
        if labels is not None:
            cluster_label_series = pd.Series(labels.values)[mask]
            if not cluster_label_series.empty:
                vc = cluster_label_series.value_counts(normalize=True)
                dominant = str(vc.index[0])
                label_breakdown = {
                    str(lbl): round(float(pct), 4)
                    for lbl, pct in vc.head(8).items()
                }

        # Network profile from original unscaled data
        network_profile = _build_network_profile(original_df, mask)

        # Security interpretation
        interpretation = _interpret_cluster(
            dominant, count, label_breakdown, network_profile
        )

        profiles.append(
            {
                "cluster_id": cid,
                "size": count,
                "dominant_attack": dominant,
                "label_breakdown": label_breakdown,
                "network_profile": network_profile,
                "security_interpretation": interpretation,
                "centroid_top_features": {
                    feature_names[i]: round(centroid[i], 4)
                    for i in np.argsort(np.abs(centroid))[::-1][:5]
                },
            }
        )

    result = {
        "k": k,
        "silhouette_score": sil,
        "clusters": profiles,
    }

    # Persist
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(km, f)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _build_network_profile(
    df: Optional[pd.DataFrame], mask: np.ndarray
) -> dict:
    """Compute network statistics for a cluster from the original DataFrame."""
    if df is None or df.empty:
        return {}

    cluster_df = df.iloc[mask.nonzero()[0]] if hasattr(mask, "nonzero") else df[mask]
    if cluster_df.empty:
        return {}

    profile: dict = {}

    # Average flow duration
    if "Flow Duration" in cluster_df.columns:
        profile["avg_duration_us"] = round(
            float(cluster_df["Flow Duration"].median()), 2
        )

    # Average packets per second
    if "Flow Packets/s" in cluster_df.columns:
        profile["avg_packets_per_sec"] = round(
            float(cluster_df["Flow Packets/s"].median()), 2
        )

    # Average flow bytes per second
    if "Flow Bytes/s" in cluster_df.columns:
        col = pd.to_numeric(cluster_df["Flow Bytes/s"], errors="coerce")
        profile["avg_flow_bytes_per_sec"] = round(float(col.median()), 2)

    # Average packet size
    if "Average Packet Size" in cluster_df.columns:
        profile["avg_packet_size"] = round(
            float(cluster_df["Average Packet Size"].median()), 2
        )

    # Top destination ports
    if "Destination Port" in cluster_df.columns:
        ports = (
            cluster_df["Destination Port"]
            .dropna()
            .astype(int)
            .value_counts()
            .head(5)
        )
        profile["top_destination_ports"] = {
            str(port): int(cnt) for port, cnt in ports.items()
        }

    # Dominant protocol indicator via TCP flags
    syn_count = 0
    if "SYN Flag Count" in cluster_df.columns:
        syn_count = int((cluster_df["SYN Flag Count"] > 0).sum())
    total = len(cluster_df)
    tcp_ratio = syn_count / max(total, 1)
    profile["tcp_ratio"] = round(tcp_ratio, 4)
    profile["dominant_protocol"] = "TCP" if tcp_ratio > 0.3 else "Mixed/UDP"

    # Average forward and backward packets
    if "Total Fwd Packets" in cluster_df.columns:
        profile["avg_fwd_packets"] = round(
            float(cluster_df["Total Fwd Packets"].median()), 2
        )
    if "Total Backward Packets" in cluster_df.columns:
        profile["avg_bwd_packets"] = round(
            float(cluster_df["Total Backward Packets"].median()), 2
        )

    return profile


def _interpret_cluster(
    dominant: str,
    size: int,
    breakdown: dict[str, float],
    network_profile: dict,
) -> str:
    """Generate a human-readable security interpretation for a cluster."""
    if dominant.upper() == "BENIGN":
        return (
            f"Normal traffic cluster ({size:,} flows). "
            f"Predominantly benign network activity."
        )

    parts = [f"{dominant} behaviour cluster ({size:,} flows)."]

    # Attack purity
    attack_pct = sum(v for k, v in breakdown.items() if k.upper() != "BENIGN")
    if attack_pct > 0.8:
        parts.append(f"Highly concentrated attack cluster ({attack_pct:.0%} attack traffic).")
    elif attack_pct > 0.5:
        parts.append(f"Mixed cluster with {attack_pct:.0%} attack traffic.")

    # Network characteristics
    if network_profile:
        dur = network_profile.get("avg_duration_us")
        pps = network_profile.get("avg_packets_per_sec")
        proto = network_profile.get("dominant_protocol", "")

        if dur is not None and dur < 1000:
            parts.append("Short-duration flows — possible scanning or flood.")
        elif dur is not None and dur > 1_000_000:
            parts.append("Long-duration flows — possible slow attack or session hijack.")

        if pps is not None and pps > 10000:
            parts.append("High packet rate — possible volumetric attack.")

        if proto:
            parts.append(f"Dominant protocol: {proto}.")

    return " ".join(parts)


def load_kmeans():
    """Load the persisted KMeans model from disk."""
    if not KMEANS_PATH.exists():
        raise FileNotFoundError("KMeans model not found. Run train_kmeans() first.")
    with open(KMEANS_PATH, "rb") as f:
        return pickle.load(f)


def get_cluster_profiles() -> dict:
    """Return the cached cluster profiles (or empty dict if not yet computed)."""
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def assign_clusters(X: np.ndarray) -> np.ndarray:
    """Predict cluster assignments for new feature matrix X.

    Requires the KMeans model to have been trained first.
    """
    km = load_kmeans()
    return km.predict(X)


def elbow_scores(X: np.ndarray, k_range: range = range(2, 12)) -> list[dict]:
    """Compute inertia for a range of k values (elbow method helper).

    Returns [{"k": int, "inertia": float}, ...]
    """
    from sklearn.cluster import KMeans

    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        km.fit(X)
        scores.append({"k": k, "inertia": float(km.inertia_)})
    return scores
