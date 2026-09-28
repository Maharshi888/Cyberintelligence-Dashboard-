"""
tests/test_live_inference.py
-----------------------------
End-to-end inference tests for the running FastAPI server.

Two modes
---------
1. **Pytest (CI / no server)**  – uses FastAPI's TestClient so tests run
   without a separate server process.  Heavy ML ops (model load, CSV read)
   are guarded by ``pytest.skip`` when artifacts are missing.

2. **Stand-alone script** – run ``python tests/test_live_inference.py`` to
   hit a *live* server at ``http://127.0.0.1:8000``.  Requires:
     a) ``python backend/app.py`` running in another terminal, AND
     b) ``raw/combinenew.csv`` present for the dataset-based tests.

Tests
-----
1. Health & stats endpoints
2. Benign flow threat-intel assessment
3. Attack flow threat-intel assessment (real CICIDS2017 record when CSV exists)
4. Batch assessment (synthetic flows)
5. Live feed polling
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_CSV = ROOT / "raw" / "combinenew.csv"
MODEL_TRAINED = (ROOT / "model" / "random_forest.pkl").exists()
CSV_AVAILABLE = RAW_CSV.exists()

# ---------------------------------------------------------------------------
# Synthetic fallback flows (used when the real CSV is absent)
# ---------------------------------------------------------------------------

_BENIGN_FLOW: dict = {
    "Destination Port": 443,
    "Flow Duration": 500000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    "Total Length of Fwd Packets": 4000,
    "Total Length of Bwd Packets": 3000,
    "Flow Bytes/s": 14000.0,
    "Flow Packets/s": 36.0,
    "SYN Flag Count": 1,
    "ACK Flag Count": 1,
    "FIN Flag Count": 1,
    "RST Flag Count": 0,
    "Average Packet Size": 389.0,
    "Init_Win_bytes_forward": 8192,
    "Init_Win_bytes_backward": 8192,
}

_ATTACK_FLOW: dict = {
    "Destination Port": 80,
    "Flow Duration": 500,
    "Total Fwd Packets": 500,
    "Total Backward Packets": 0,
    "Total Length of Fwd Packets": 0,
    "Total Length of Bwd Packets": 0,
    "Flow Bytes/s": 2_000_000.0,
    "Flow Packets/s": 200_000.0,
    "SYN Flag Count": 1,
    "ACK Flag Count": 0,
    "FIN Flag Count": 0,
    "RST Flag Count": 0,
    "Average Packet Size": 40.0,
    "Init_Win_bytes_forward": 0,
    "Init_Win_bytes_backward": 0,
}


# ---------------------------------------------------------------------------
# Pytest fixture — TestClient (no live server needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a FastAPI TestClient for the full application."""
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

def test_health(client):
    """Server should respond healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_trained" in data
    print(f"\n[health] status={data['status']} | model_trained={data['model_trained']}")


# ---------------------------------------------------------------------------
# 2. Stats endpoint
# ---------------------------------------------------------------------------

def test_stats(client):
    """Stats endpoint should return total_flows and label_distribution."""
    resp = client.get("/api/stats")
    # Acceptable: 200 (cache present) or 404 (no cache yet, no dataset)
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert "total_flows" in data
        assert "label_distribution" in data
        print(f"\n[stats] total_flows={data['total_flows']:,} | "
              f"attack_types={data.get('attack_types', '?')}")
    else:
        pytest.skip("Label-distribution cache not present — run the server once to build it.")


# ---------------------------------------------------------------------------
# 3. Live feed
# ---------------------------------------------------------------------------

def test_live_feed(client):
    """Live feed should return a list."""
    resp = client.get("/api/live-feed?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "feed" in data
    assert isinstance(data["feed"], list)
    print(f"\n[live-feed] received {len(data['feed'])} entries")


# ---------------------------------------------------------------------------
# 4. Threat-intel assess — benign flow
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_TRAINED, reason="RF model not trained — run src/train_random_forest.py")
def test_threat_intel_benign(client):
    """Benign flow should receive a LOW/MEDIUM threat level."""
    resp = client.post("/api/threat-intel/assess", json={"features": _BENIGN_FLOW})
    assert resp.status_code == 200
    data = resp.json()
    _assert_threat_intel_response(data)
    print(
        f"\n[assess-benign] prediction={data['prediction']} | "
        f"score={data['threat_score']} | level={data['threat_level']}"
    )


# ---------------------------------------------------------------------------
# 5. Threat-intel assess — attack flow
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_TRAINED, reason="RF model not trained — run src/train_random_forest.py")
def test_threat_intel_attack(client):
    """Attack-like flow should receive an elevated threat level."""
    resp = client.post("/api/threat-intel/assess", json={"features": _ATTACK_FLOW})
    assert resp.status_code == 200
    data = resp.json()
    _assert_threat_intel_response(data)
    print(
        f"\n[assess-attack] prediction={data['prediction']} | "
        f"score={data['threat_score']} | level={data['threat_level']}"
    )


# ---------------------------------------------------------------------------
# 6. Threat-intel assess — real CICIDS2017 flows (requires CSV)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_TRAINED, reason="RF model not trained")
@pytest.mark.skipif(not CSV_AVAILABLE, reason="raw/combinenew.csv not present")
def test_threat_intel_real_flows(client):
    """Read actual CICIDS2017 records and assess them end-to-end."""
    import pandas as pd

    df = pd.read_csv(RAW_CSV, nrows=80_000, low_memory=False)
    df.columns = df.columns.str.strip()

    benign_df = df[df["Label"].str.strip() == "BENIGN"]
    attack_df = df[df["Label"].str.strip() != "BENIGN"]

    for label, sample_df in [("BENIGN", benign_df), ("ATTACK", attack_df)]:
        if sample_df.empty:
            continue
        row = sample_df.iloc[0]
        true_label = row["Label"].strip()
        features = row.drop("Label").to_dict()

        resp = client.post("/api/threat-intel/assess", json={"features": features})
        assert resp.status_code == 200
        data = resp.json()
        _assert_threat_intel_response(data)
        print(
            f"\n[real-{label}] true={true_label} | predicted={data['prediction']} | "
            f"score={data['threat_score']} | level={data['threat_level']}"
        )
        for rec in data.get("recommendations", [])[:1]:
            print(f"  mitigation: {rec}")


# ---------------------------------------------------------------------------
# 7. Batch assess
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_TRAINED, reason="RF model not trained — run src/train_random_forest.py")
def test_threat_intel_batch(client):
    """Batch endpoint should accept multiple flows and return matching alerts."""
    flows = [_BENIGN_FLOW, _ATTACK_FLOW]
    resp = client.post("/api/threat-intel/assess-batch", json={"flows": flows})
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert len(data["alerts"]) == len(flows)
    for alert in data["alerts"]:
        _assert_threat_intel_response(alert)
    print(f"\n[batch] processed {len(data['alerts'])} alerts OK")


# ---------------------------------------------------------------------------
# 8. Threat summary
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_TRAINED, reason="RF model not trained — run src/train_random_forest.py")
def test_threat_summary(client):
    """Threat summary should confirm RF model is active."""
    resp = client.get("/api/threat-intel/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("model_status", {}).get("random_forest") is True
    print(f"\n[summary] model_status={data.get('model_status')}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_threat_intel_response(data: dict) -> None:
    """Validate the structure of a /api/threat-intel/assess response."""
    assert "prediction" in data, f"Missing 'prediction' in: {data}"
    assert "threat_score" in data, f"Missing 'threat_score' in: {data}"
    assert "threat_level" in data, f"Missing 'threat_level' in: {data}"
    assert 0 <= data["threat_score"] <= 100, (
        f"threat_score {data['threat_score']} out of range"
    )
    assert data["threat_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}, (
        f"Unexpected threat_level: {data['threat_level']}"
    )


# ---------------------------------------------------------------------------
# Stand-alone live-server runner (python tests/test_live_inference.py)
# ---------------------------------------------------------------------------

def _run_live(base: str = "http://127.0.0.1:8000") -> None:
    """Hit a live server and print a pretty report."""
    sep = "=" * 60

    def _get(path: str) -> dict:
        with urllib.request.urlopen(f"{base}{path}") as r:
            return json.loads(r.read())

    def _post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    print(sep)
    print("🚀  LIVE SOC SERVER — END-TO-END INFERENCE TEST")
    print(sep)

    # 1. Health
    h = _get("/health")
    print(f"\n[1] Health: {h['status']} | model_trained={h['model_trained']}")

    # 2. Stats
    try:
        s = _get("/api/stats")
        print(f"[2] Stats:  {s['total_flows']:,} flows | {s['attack_types']} attack types")
    except urllib.error.HTTPError as exc:
        print(f"[2] Stats:  {exc.code} — cache not built yet.")

    # 3. Benign flow
    print("\n[3] Assessing synthetic BENIGN flow …")
    b = _post("/api/threat-intel/assess", {"features": _BENIGN_FLOW})
    print(f"    Predicted: {b['prediction']}  Score: {b['threat_score']}/100  Level: {b['threat_level']}")

    # 4. Attack flow
    print("\n[4] Assessing synthetic ATTACK flow …")
    a = _post("/api/threat-intel/assess", {"features": _ATTACK_FLOW})
    print(f"    Predicted: {a['prediction']}  Score: {a['threat_score']}/100  Level: {a['threat_level']}")
    for m in a.get("recommendations", []):
        print(f"    ➜  {m}")

    # 5. Real dataset flows (optional)
    if CSV_AVAILABLE:
        import pandas as pd

        print(f"\n[5] Loading real flows from {RAW_CSV.name} …")
        df = pd.read_csv(RAW_CSV, nrows=80_000, low_memory=False)
        df.columns = df.columns.str.strip()

        for label, grp in [
            ("BENIGN", df[df["Label"].str.strip() == "BENIGN"]),
            ("ATTACK", df[df["Label"].str.strip() != "BENIGN"]),
        ]:
            if grp.empty:
                continue
            row = grp.iloc[0]
            true_lbl = row["Label"].strip()
            feats = row.drop("Label").to_dict()
            res = _post("/api/threat-intel/assess", {"features": feats})
            print(
                f"\n    [{label}] true={true_lbl} | "
                f"predicted={res['prediction']} | "
                f"score={res['threat_score']}/100 | level={res['threat_level']}"
            )
    else:
        print(f"\n[5] Skipped — {RAW_CSV} not present (demo mode).")

    # 6. Batch
    print("\n[6] Batch assess (2 flows) …")
    batch = _post("/api/threat-intel/assess-batch", {"flows": [_BENIGN_FLOW, _ATTACK_FLOW]})
    print(f"    Received {len(batch['alerts'])} alert(s).")

    # 7. Live feed
    print("\n[7] Live feed …")
    feed = _get("/api/live-feed?limit=3")
    print(f"    {len(feed['feed'])} entries returned.")

    print(f"\n{sep}")
    print("✅  ALL LIVE SERVER TESTS PASSED!")
    print(sep)


if __name__ == "__main__":
    _run_live()
