"""
src/models/threat_intel.py
----------------------------
Threat Intelligence Engine — combines Random Forest classification,
K-Means clustering, and Apriori association rule mining into a unified
threat assessment pipeline.

Usage:
    from src.models.threat_intel import ThreatIntelligenceEngine

    engine = ThreatIntelligenceEngine()
    alert = engine.assess_flow(flow_features_dict)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """A matched Apriori association rule."""
    antecedent: list[str]
    consequent: list[str]
    support: float
    confidence: float
    lift: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThreatAlert:
    """Unified threat intelligence assessment for a single network flow."""
    # RF classification
    prediction: str
    confidence: float
    probabilities: dict[str, float]

    # Severity assessment
    severity: str           # LOW, MEDIUM, HIGH, CRITICAL
    severity_score: float   # 0.0 – 1.0

    # K-Means cluster context
    cluster_id: int
    cluster_profile_name: str
    cluster_size: int
    cluster_label_breakdown: dict[str, float]

    # Apriori pattern matches
    matched_patterns: list[PatternMatch]

    # Feature contributions
    top_contributing_features: list[dict]

    # Human-readable explanation
    explanation: str

    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matched_patterns"] = [p.to_dict() if isinstance(p, PatternMatch) else p for p in self.matched_patterns]
        d["threat_score"] = round(self.severity_score * 100, 1)
        d["threat_level"] = self.severity
        d["recommendations"] = self.get_recommendations()
        return d

    def get_recommendations(self) -> list[str]:
        """Generate actionable SOC mitigation steps based on threat assessment."""
        recs = []
        pred = self.prediction.upper()
        if pred == "BENIGN" or self.severity == "LOW":
            recs.append("Traffic pattern matches baseline benign profile. No blocking action required.")
            recs.append("Continue standard telemetry collection and logging.")
        elif "DOS" in pred or "DDOS" in pred:
            recs.append("Trigger automated upstream rate-limiting and SYN flood scrubbing.")
            recs.append("Deploy stateful connection rate caps on the targeted destination port.")
            recs.append("Enable IP reputation filtering and block top anomalous subnet emitters.")
        elif "SCAN" in pred or "PORTSCAN" in pred:
            recs.append("Isolate scanner IP source at boundary firewall.")
            recs.append("Block port sweeps and disable non-essential public-facing listening ports.")
        elif "PATATOR" in pred or "BRUTE FORCE" in pred:
            recs.append("Enforce multi-factor authentication (MFA) and lock compromised accounts.")
            recs.append("Enable fail2ban brute-force protection with temporary IP drop rules.")
        elif "BOT" in pred or "INFILTRATION" in pred:
            recs.append("Immediate incident response: isolate affected endpoint from internal LAN.")
            recs.append("Inspect outbound DNS/C2 beaconing channels and sinkhole malicious domains.")
        else:
            recs.append("Elevated risk detected. Quarantine suspect flow and notify SOC tier 2.")
            recs.append("Review full packet payload and firewall access logs.")
        return recs



# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

ATTACK_BASE_SEVERITY = {
    "BENIGN":                       0.0,
    "FTP-Patator":                  0.55,
    "SSH-Patator":                  0.55,
    "DoS slowloris":                0.65,
    "DoS Slowhttptest":             0.65,
    "DoS GoldenEye":                0.70,
    "DoS Hulk":                     0.75,
    "Heartbleed":                   0.90,
    "Web Attack \u2013 Brute Force":     0.60,
    "Web Attack \u2013 XSS":             0.65,
    "Web Attack \u2013 Sql Injection":    0.80,
    "Infiltration":                 0.85,
    "Bot":                          0.80,
    "PortScan":                     0.50,
    "DDoS":                         0.85,
}


def _severity_label(score: float) -> str:
    """Convert a 0–1 severity score to a categorical label."""
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ThreatIntelligenceEngine:
    """Combines RF, K-Means, and Apriori into a unified threat assessment.

    The engine is designed to be instantiated once (e.g. at server startup)
    and reused for every incoming flow.  All three sub-models are optional —
    the engine gracefully degrades if K-Means or Apriori haven't been run.
    """

    def __init__(self) -> None:
        # RF bundle — required
        from src.models.random_forest import load_bundle, is_trained
        if not is_trained():
            raise FileNotFoundError(
                "RF model not trained. Run `python src/train_random_forest.py` first."
            )
        self.rf_bundle = load_bundle()
        self._model, self._scaler, self._encoder, self._feature_columns = self.rf_bundle

        # K-Means — optional
        self._kmeans = None
        self._cluster_profiles: dict = {}
        try:
            from src.models.kmeans import load_kmeans, get_cluster_profiles
            self._kmeans = load_kmeans()
            self._cluster_profiles = get_cluster_profiles()
        except Exception:
            pass

        # Apriori rules — optional
        self._apriori_rules: list[dict] = []
        try:
            from src.models.apriori import get_top_rules
            self._apriori_rules = get_top_rules(n=200)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_flow(self, flow: dict) -> ThreatAlert:
        """Perform a full threat intelligence assessment on a single flow.

        Args:
            flow: Dict mapping feature names to numeric values.

        Returns:
            A ThreatAlert with combined intelligence from all three models.
        """
        # 1. RF prediction
        rf_result = self._predict_rf(flow)

        # 2. Feature contributions
        top_features = self._compute_feature_contributions(flow)

        # 3. K-Means cluster assignment
        cluster_id, cluster_profile_name, cluster_size, cluster_breakdown = (
            self._assign_cluster(flow)
        )

        # 4. Apriori pattern matching
        matched_patterns = self._match_patterns(flow, rf_result["prediction"])

        # 5. Compute composite severity
        severity_score = self._compute_severity(
            rf_result, cluster_breakdown, matched_patterns
        )
        severity = _severity_label(severity_score)

        # 6. Generate human-readable explanation
        explanation = self._generate_explanation(
            rf_result, severity, cluster_id, cluster_profile_name,
            matched_patterns, top_features,
        )

        return ThreatAlert(
            prediction=rf_result["prediction"],
            confidence=rf_result["confidence"],
            probabilities=rf_result["probabilities"],
            severity=severity,
            severity_score=round(severity_score, 4),
            cluster_id=cluster_id,
            cluster_profile_name=cluster_profile_name,
            cluster_size=cluster_size,
            cluster_label_breakdown=cluster_breakdown,
            matched_patterns=matched_patterns,
            top_contributing_features=top_features,
            explanation=explanation,
        )

    def assess_batch(self, flows: list[dict]) -> list[ThreatAlert]:
        """Assess multiple flows."""
        return [self.assess_flow(f) for f in flows]

    def get_threat_summary(self) -> dict:
        """Return a high-level threat landscape summary based on loaded models."""
        clusters = self._cluster_profiles.get("clusters", [])
        attack_clusters = [
            c for c in clusters
            if c.get("dominant_attack", "").upper() != "BENIGN"
        ]

        return {
            "model_status": {
                "random_forest": True,
                "kmeans": self._kmeans is not None,
                "apriori": len(self._apriori_rules) > 0,
            },
            "attack_classes": self._encoder.classes_.tolist(),
            "num_clusters": len(clusters),
            "attack_clusters": len(attack_clusters),
            "num_apriori_rules": len(self._apriori_rules),
            "silhouette_score": self._cluster_profiles.get("silhouette_score"),
            "top_attack_patterns": self._apriori_rules[:5],
        }

    # ------------------------------------------------------------------
    # Internal: RF
    # ------------------------------------------------------------------

    def _predict_rf(self, flow: dict) -> dict:
        """Classify using Random Forest."""
        from src.models.random_forest import predict_single
        return predict_single(flow, self.rf_bundle)

    def _compute_feature_contributions(self, flow: dict, top_n: int = 5) -> list[dict]:
        """Estimate which features contributed most to this specific prediction.

        Uses feature importance × feature value deviation from training mean
        as a simple proxy for per-sample contribution.
        """
        importances = self._model.feature_importances_
        scaler_means = self._scaler.mean_
        scaler_scales = self._scaler.scale_

        contributions = []
        for i, col in enumerate(self._feature_columns):
            raw_val = float(flow.get(col, 0))
            # Normalised deviation from training mean
            if scaler_scales[i] > 0:
                deviation = abs((raw_val - scaler_means[i]) / scaler_scales[i])
            else:
                deviation = 0.0
            score = float(importances[i]) * deviation
            contributions.append({
                "feature": col,
                "importance": round(float(importances[i]), 6),
                "value": raw_val,
                "contribution_score": round(score, 6),
            })

        contributions.sort(key=lambda x: x["contribution_score"], reverse=True)
        return contributions[:top_n]

    # ------------------------------------------------------------------
    # Internal: K-Means
    # ------------------------------------------------------------------

    def _assign_cluster(self, flow: dict) -> tuple[int, str, int, dict]:
        """Assign the flow to a K-Means cluster and return profile info.

        Returns:
            (cluster_id, profile_name, cluster_size, label_breakdown)
        """
        if self._kmeans is None:
            return -1, "Unknown", 0, {}

        # Build feature vector
        df = pd.DataFrame([flow])
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[self._feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = self._scaler.transform(df)

        cluster_id = int(self._kmeans.predict(scaled)[0])

        # Look up profile
        clusters = self._cluster_profiles.get("clusters", [])
        profile = next(
            (c for c in clusters if c["cluster_id"] == cluster_id), None
        )
        if profile:
            profile_name = profile.get("dominant_attack", "Unknown")
            cluster_size = profile.get("size", 0)
            label_breakdown = profile.get("label_breakdown", {})
        else:
            profile_name = "Unknown"
            cluster_size = 0
            label_breakdown = {}

        return cluster_id, profile_name, cluster_size, label_breakdown

    # ------------------------------------------------------------------
    # Internal: Apriori pattern matching
    # ------------------------------------------------------------------

    def _match_patterns(
        self, flow: dict, prediction: str, max_matches: int = 5
    ) -> list[PatternMatch]:
        """Check which Apriori rules match the characteristics of this flow."""
        if not self._apriori_rules:
            return []

        # Build the flow's item set (same logic as apriori.py transaction builder)
        items = self._flow_to_items(flow, prediction)

        matched: list[PatternMatch] = []
        for rule in self._apriori_rules:
            ante = set(rule["antecedent"])
            cons = set(rule["consequent"])
            # A rule matches if all antecedent items are present in the flow
            if ante.issubset(items):
                matched.append(PatternMatch(
                    antecedent=rule["antecedent"],
                    consequent=rule["consequent"],
                    support=rule["support"],
                    confidence=rule["confidence"],
                    lift=rule["lift"],
                ))
                if len(matched) >= max_matches:
                    break

        # Sort by lift descending
        matched.sort(key=lambda p: p.lift, reverse=True)
        return matched

    def _flow_to_items(self, flow: dict, prediction: str) -> set[str]:
        """Convert flow features to a set of Apriori-style items for matching."""
        items: set[str] = set()

        # Attack label
        items.add(f"ATTACK={prediction}")

        # Destination port bucket
        from src.models.apriori import PORT_BUCKETS
        port = flow.get("Destination Port", -1)
        if port is not None and not pd.isna(port):
            port = int(port)
            matched_port = False
            for name, ports in PORT_BUCKETS.items():
                if port in ports:
                    items.add(f"PORT={name}")
                    matched_port = True
                    break
            if not matched_port and port > 0:
                items.add("PORT=OTHER")

        # TCP flags
        for flag_col, label in [
            ("SYN Flag Count", "SYN_SET"),
            ("ACK Flag Count", "ACK_SET"),
            ("FIN Flag Count", "FIN_SET"),
            ("RST Flag Count", "RST_SET"),
        ]:
            if flow.get(flag_col, 0) and flow[flag_col] > 0:
                items.add(label)

        # Packet size category
        avg_size = flow.get("Average Packet Size", 0)
        if avg_size and not pd.isna(avg_size):
            if avg_size < 100:
                items.add("PKT_SIZE=SMALL")
            elif avg_size < 1000:
                items.add("PKT_SIZE=MEDIUM")
            else:
                items.add("PKT_SIZE=LARGE")

        # Flow rate categories
        pps = flow.get("Flow Packets/s", 0)
        if pps and not pd.isna(pps):
            if pps > 10000:
                items.add("RATE=HIGH_PPS")
            elif pps < 100:
                items.add("RATE=LOW_PPS")

        # Duration categories
        duration = flow.get("Flow Duration", 0)
        if duration is not None and not pd.isna(duration):
            if duration < 1000:
                items.add("DURATION=SHORT")
            elif duration < 100000:
                items.add("DURATION=MEDIUM")
            else:
                items.add("DURATION=LONG")

        return items

    # ------------------------------------------------------------------
    # Internal: Severity computation
    # ------------------------------------------------------------------

    def _compute_severity(
        self,
        rf_result: dict,
        cluster_breakdown: dict,
        matched_patterns: list[PatternMatch],
    ) -> float:
        """Compute a composite severity score (0–1) from all three models.

        Weighting:
            40%  RF confidence × base attack severity
            30%  Cluster attack dominance ratio
            30%  Apriori pattern match strength (max lift, normalised)
        """
        prediction = rf_result["prediction"]
        confidence = rf_result["confidence"]

        # --- RF signal (40%) ---
        base_sev = ATTACK_BASE_SEVERITY.get(prediction, 0.5)
        rf_signal = base_sev * confidence

        # --- Cluster signal (30%) ---
        cluster_signal = 0.0
        if cluster_breakdown:
            # Ratio of attack flows in the assigned cluster
            attack_ratio = sum(
                v for k, v in cluster_breakdown.items()
                if k.upper() != "BENIGN"
            )
            cluster_signal = min(attack_ratio, 1.0)

        # --- Apriori signal (30%) ---
        apriori_signal = 0.0
        if matched_patterns:
            max_lift = max(p.lift for p in matched_patterns)
            # Normalise lift: lift of 1 = neutral, lift > 3 = strong signal
            apriori_signal = min(max_lift / 5.0, 1.0)

        composite = (0.40 * rf_signal) + (0.30 * cluster_signal) + (0.30 * apriori_signal)
        return min(max(composite, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Internal: Explanation generation
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        rf_result: dict,
        severity: str,
        cluster_id: int,
        cluster_name: str,
        patterns: list[PatternMatch],
        top_features: list[dict],
    ) -> str:
        """Generate a human-readable threat intelligence explanation."""
        prediction = rf_result["prediction"]
        confidence = rf_result["confidence"]

        if prediction.upper() == "BENIGN":
            return (
                f"Traffic classified as BENIGN with {confidence:.0%} confidence. "
                f"No threat indicators detected."
            )

        parts = [
            f"⚠️ {severity} SEVERITY — {prediction} detected "
            f"with {confidence:.0%} confidence.",
        ]

        # Cluster context
        if cluster_id >= 0:
            parts.append(
                f"Flow assigned to Cluster #{cluster_id} "
                f"(dominant behaviour: {cluster_name})."
            )

        # Pattern matches
        if patterns:
            top_pattern = patterns[0]
            ante_str = " + ".join(top_pattern.antecedent)
            cons_str = ", ".join(top_pattern.consequent)
            parts.append(
                f"Matches attack pattern: [{ante_str}] → [{cons_str}] "
                f"(lift: {top_pattern.lift:.2f}×)."
            )

        # Key features
        if top_features:
            feat_names = [f["feature"] for f in top_features[:3]]
            parts.append(
                f"Key contributing features: {', '.join(feat_names)}."
            )

        return " ".join(parts)
