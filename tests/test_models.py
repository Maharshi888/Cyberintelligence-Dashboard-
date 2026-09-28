"""
tests/test_models.py
---------------------
Unit tests for data loading, preprocessing, feature engineering,
and model modules. Uses a synthetic 200-row dataset.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """200-row synthetic CICIDS-like DataFrame."""
    rng = np.random.default_rng(42)
    n = 200
    labels = rng.choice(["BENIGN", "DDoS", "PortScan", "Bot"], size=n, p=[0.6, 0.2, 0.15, 0.05])
    data = {
        "Destination Port":            rng.integers(1, 65535, size=n),
        "Flow Duration":               rng.integers(0, 1_000_000, size=n),
        "Total Fwd Packets":           rng.integers(1, 100, size=n),
        "Total Backward Packets":      rng.integers(0, 100, size=n),
        "Total Length of Fwd Packets": rng.integers(0, 10000, size=n),
        "Total Length of Bwd Packets": rng.integers(0, 10000, size=n),
        "Flow Bytes/s":                rng.uniform(0, 5_000_000, size=n),
        "Flow Packets/s":              rng.uniform(0, 1_000_000, size=n),
        "SYN Flag Count":              rng.integers(0, 2, size=n),
        "ACK Flag Count":              rng.integers(0, 2, size=n),
        "FIN Flag Count":              rng.integers(0, 2, size=n),
        "RST Flag Count":              rng.integers(0, 2, size=n),
        "Average Packet Size":         rng.uniform(0, 2000, size=n),
        "Init_Win_bytes_forward":      rng.integers(-1, 65535, size=n),
        "Init_Win_bytes_backward":     rng.integers(-1, 65535, size=n),
        "Label":                       labels,
    }
    return pd.DataFrame(data)


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_clean_dataframe_removes_inf(self, sample_df):
        from src.data.preprocessing import clean_dataframe

        # Cast to float first — pandas 2.x won't accept inf in int64 columns
        sample_df["Destination Port"] = sample_df["Destination Port"].astype(float)
        sample_df.iloc[0, 0] = np.inf
        sample_df.iloc[1, 0] = -np.inf
        cleaned = clean_dataframe(sample_df, feature_cols=["Destination Port"])
        assert not np.isinf(cleaned["Destination Port"]).any()

    def test_clean_dataframe_fills_nan(self, sample_df):
        from src.data.preprocessing import clean_dataframe

        sample_df.iloc[0, 0] = np.nan
        cleaned = clean_dataframe(sample_df, feature_cols=["Destination Port"])
        assert not cleaned["Destination Port"].isna().any()

    def test_encode_labels(self, sample_df):
        from src.data.preprocessing import encode_labels

        encoded, encoder = encode_labels(sample_df["Label"])
        assert len(encoded) == len(sample_df)
        assert set(encoder.classes_).issubset({"BENIGN", "DDoS", "PortScan", "Bot"})

    def test_scale_features(self, sample_df):
        from src.data.preprocessing import scale_features

        numeric = sample_df.select_dtypes(include=np.number)
        X_train = numeric.iloc[:150]
        X_test  = numeric.iloc[150:]
        X_tr_scaled, X_te_scaled, scaler = scale_features(X_train, X_test)
        assert X_tr_scaled.shape == X_train.shape
        assert X_te_scaled.shape == X_test.shape
        # Mean of scaled training should be ~0
        assert abs(X_tr_scaled.mean()) < 1.0


# ── Feature Engineering ───────────────────────────────────────────────────────

class TestEngineering:
    def test_flag_ratios_added(self, sample_df):
        from src.features.engineering import compute_flag_ratios

        out = compute_flag_ratios(sample_df)
        assert "SYN Flag Count Ratio" in out.columns

    def test_flag_ratios_no_div_zero(self, sample_df):
        from src.features.engineering import compute_flag_ratios

        sample_df["Total Fwd Packets"] = 0
        sample_df["Total Backward Packets"] = 0
        out = compute_flag_ratios(sample_df)
        assert not out.filter(like="Ratio").isin([np.inf, -np.inf]).any().any()


# ── K-Means ───────────────────────────────────────────────────────────────────

class TestKMeans:
    def test_train_kmeans_returns_profiles(self, sample_df, tmp_path, monkeypatch):
        from src.models import kmeans as km_mod

        monkeypatch.setattr(km_mod, "KMEANS_PATH",   tmp_path / "kmeans.pkl")
        monkeypatch.setattr(km_mod, "PROFILES_PATH", tmp_path / "cluster_profiles.json")

        from src.data.preprocessing import clean_dataframe
        numeric = [c for c in sample_df.select_dtypes(include=np.number).columns if c != "Label"]
        df_clean = clean_dataframe(sample_df, feature_cols=numeric)
        X = df_clean[numeric].values

        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        result = km_mod.train_kmeans(X_scaled, feature_names=numeric, labels=sample_df["Label"], k=3)
        assert result["k"] == 3
        assert len(result["clusters"]) == 3
        assert "silhouette_score" in result


# ── Apriori ───────────────────────────────────────────────────────────────────

class TestApriori:
    def test_mine_patterns_returns_rules(self, sample_df, tmp_path, monkeypatch):
        pytest.importorskip("mlxtend")
        from src.models import apriori as apr_mod

        monkeypatch.setattr(apr_mod, "RULES_PATH", tmp_path / "apriori_rules.json")

        rules = apr_mod.mine_patterns(sample_df, min_support=0.05, min_confidence=0.5)
        assert isinstance(rules, list)
        if rules:
            assert "antecedent" in rules[0]
            assert "lift" in rules[0]

    def test_get_top_rules_empty_if_not_mined(self, tmp_path, monkeypatch):
        from src.models import apriori as apr_mod

        monkeypatch.setattr(apr_mod, "RULES_PATH", tmp_path / "nonexistent.json")
        assert apr_mod.get_top_rules() == []


# ── Evaluation Metrics ────────────────────────────────────────────────────────

class TestMetrics:
    def test_classification_summary(self):
        from src.evaluation.metrics import classification_summary

        y_true = np.array([0, 1, 0, 1, 2])
        y_pred = np.array([0, 1, 0, 0, 2])
        classes = ["BENIGN", "DDoS", "PortScan"]
        result = classification_summary(y_true, y_pred, classes)
        assert 0 <= result["accuracy"] <= 1
        assert "per_class" in result

    def test_cluster_silhouette_valid(self):
        from src.evaluation.metrics import cluster_silhouette

        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        labels = rng.integers(0, 3, size=100)
        score = cluster_silhouette(X, labels)
        assert -1.0 <= score <= 1.0

    def test_cluster_silhouette_degenerate(self):
        from src.evaluation.metrics import cluster_silhouette

        X = np.ones((50, 3))
        labels = np.zeros(50, dtype=int)  # all same cluster
        assert cluster_silhouette(X, labels) == -1.0
