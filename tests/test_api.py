"""
tests/test_api.py
------------------
Integration tests for the FastAPI endpoints using TestClient.
These tests mock heavy ML operations to keep the test suite fast.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient; startup cache/model loads are no-ops when files are absent."""
    from fastapi.testclient import TestClient
    from backend.app import app

    # Use the lifespan context so the app initialises properly during tests.
    # Heavy operations (model load, CSV load) will silently skip if files are absent.
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── /api/stats ────────────────────────────────────────────────────────────────

class TestStatsEndpoint:
    def test_stats_not_found_without_cache(self, client):
        """Should return 404 when processed data doesn't exist."""
        from fastapi import HTTPException
        with patch("backend.routers.analytics._read_json", side_effect=HTTPException(status_code=404, detail="not found")):
            res = client.get("/api/stats")
            assert res.status_code == 404

    def test_stats_returns_expected_keys(self, client, tmp_path, monkeypatch):
        """With a mocked cache file, should return correct structure."""
        fake_dist = {"BENIGN": 1000, "DDoS": 200}
        with patch("backend.routers.analytics._read_json", return_value=fake_dist):
            res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_flows" in data
        assert "label_distribution" in data


# ── /api/predict ─────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_no_model_returns_503(self, client):
        with patch("src.models.random_forest.is_trained", return_value=False):
            res = client.post("/api/predict", json={"features": {"Destination Port": 80}})
        assert res.status_code == 503

    def test_predict_with_mock_model(self, client):
        mock_result = {
            "prediction": "BENIGN",
            "confidence": 0.98,
            "probabilities": {"BENIGN": 0.98, "DDoS": 0.02},
        }
        with patch("src.models.random_forest.is_trained", return_value=True), \
             patch("src.models.random_forest.load_bundle", return_value=MagicMock()), \
             patch("src.models.random_forest.predict_single", return_value=mock_result):
            res = client.post("/api/predict", json={"features": {"Destination Port": 80}})
        assert res.status_code == 200
        data = res.json()
        assert data["prediction"] == "BENIGN"
        assert 0 <= data["confidence"] <= 1


# ── /api/clusters ─────────────────────────────────────────────────────────────

class TestClustersEndpoint:
    def test_clusters_not_computed(self, client):
        with patch("src.models.kmeans.get_cluster_profiles", return_value={}):
            res = client.get("/api/clusters")
        assert res.status_code == 200
        assert res.json()["status"] == "not_computed"

    def test_clusters_train_triggers_background(self, client):
        with patch("backend.routers.clusters._run_clustering"):
            res = client.post("/api/clusters/train?k=4&sample_size=1000")
        assert res.status_code == 200
        assert res.json()["status"] in ("running", "done")

    def test_clusters_status(self, client):
        res = client.get("/api/clusters/status")
        assert res.status_code == 200
        assert "status" in res.json()


# ── /api/patterns ─────────────────────────────────────────────────────────────

class TestPatternsEndpoint:
    def test_patterns_not_computed(self, client):
        with patch("src.models.apriori.get_top_rules", return_value=[]):
            res = client.get("/api/patterns")
        assert res.status_code == 200
        assert res.json()["status"] == "not_computed"

    def test_patterns_mine_triggers_background(self, client):
        with patch("backend.routers.patterns._run_mining"):
            res = client.post("/api/patterns/mine")
        assert res.status_code == 200

    def test_patterns_status(self, client):
        res = client.get("/api/patterns/status")
        assert res.status_code == 200


# ── /api/live-feed ────────────────────────────────────────────────────────────

class TestLiveFeedEndpoint:
    def test_live_feed_returns_list(self, client):
        with patch("src.data.loader.get_label_distribution",
                   return_value={"BENIGN": 100, "DDoS": 50}):
            res = client.get("/api/live-feed?limit=5")
        assert res.status_code == 200
        data = res.json()
        assert "feed" in data
        assert isinstance(data["feed"], list)


# ── /api/threat-intel ─────────────────────────────────────────────────────────

class TestThreatIntelEndpoint:
    def test_threat_intel_summary_no_model(self, client):
        with patch("backend.routers.threat_intel._engine", None), \
             patch("src.models.threat_intel.ThreatIntelligenceEngine", side_effect=FileNotFoundError("Model files missing")):
            res = client.get("/api/threat-intel/summary")
            assert res.status_code == 503

    def test_threat_intel_summary_with_mock_engine(self, client):
        mock_engine = MagicMock()
        mock_engine.get_threat_summary.return_value = {
            "platform_status": "ONLINE",
            "models_loaded": ["Random Forest", "K-Means", "Apriori"],
        }
        with patch("backend.routers.threat_intel._get_engine", return_value=mock_engine):
            res = client.get("/api/threat-intel/summary")
            assert res.status_code == 200
            data = res.json()
            assert data["platform_status"] == "ONLINE"

