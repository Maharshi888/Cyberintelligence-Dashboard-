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

Fixes applied
-------------
#2  Cluster term is now guarded — only contributes when label_breakdown is
    non-empty (i.e. K-Means was retrained with labels).
#3  RF signal = sum(p_i * severity_i for all attack classes), replacing the
    broken base_sev * max_proba formula that gave 0 for BENIGN predictions
    regardless of residual attack probability.
#5  Apriori is demoted to an *explanation* signal only. It no longer
    contributes to the numeric score (no more +30 pts for benign DNS traffic).
#6  _flow_to_items() in the serving path now mirrors the training-time
    transaction builder (shared items: flags, port, rate, size, duration).
    PAYLOAD=ZERO and WIN_SIZE=* are also emitted here.
#7  ATTACK= items are NEVER inserted at serve time to prevent the circular
    feedback loop (RF prediction feeding back through Apriori signal).
#8  Recommendations are derived from the full probability vector and
    feature_coverage, not just the prediction string and severity bucket.
    An "uncertain" branch fires when coverage < 50% or top-2 proba are close.
#12 ATTACK_BASE_SEVERITY keys use encoder class names (underscore, not en-dash).
#13 _compute_feature_contributions excludes features not supplied by the user.
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

    # Input coverage (Fix #1/#13)
    feature_coverage: float     # 0-1 fraction of 77 features supplied
    supplied_features: int
    total_features: int
    missing_features: list[str]

    # Severity assessment
    severity: str           # LOW, MEDIUM, HIGH, CRITICAL
    severity_score: float   # 0.0 – 1.0

    # K-Means cluster context
    cluster_id: int
    cluster_profile_name: str
    cluster_size: int
    cluster_label_breakdown: dict[str, float]

    # Apriori pattern matches (explanation only — Fix #5)
    matched_patterns: list[PatternMatch]

    # Feature contributions (only supplied features — Fix #13)
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
        """Generate actionable SOC mitigation steps.

        Fix #8 — recommendations derived from probability vector + coverage,
        not just the prediction string + severity bucket.

        Decision branches:
          UNCERTAIN  : coverage < 50% OR top-2 classes within 10 pp
          BENIGN     : prediction BENIGN AND confident AND coverage >= 50%
          ATTACK     : specific attack prediction, high coverage, confident
        """
        recs = []
        pred = self.prediction.upper()
        coverage = self.feature_coverage
        confidence = self.confidence

        # --- Fix #8: uncertain branch ---
        probs = sorted(self.probabilities.values(), reverse=True)
        top1 = probs[0] if probs else 0.0
        top2 = probs[1] if len(probs) > 1 else 0.0
        uncertain = (coverage < 0.50) or (top1 - top2 < 0.10 and confidence < 0.80)

        if uncertain:
            recs.append(
                "⚠️ Prediction is uncertain: only {}/{} features supplied ({:.0%} coverage). "
                "Treat result as advisory only — submit a full 77-feature flow for a reliable assessment.".format(
                    self.supplied_features, self.total_features, coverage
                )
            )
            if pred != "BENIGN":
                recs.append("Cautionary hold: log the flow and escalate to SOC analyst for manual review.")
            else:
                recs.append("Continue standard telemetry collection. Supply full flow features to rule out attack.")
            return recs

        # --- Confident benign ---
        if pred == "BENIGN":
            recs.append("Traffic pattern matches baseline benign profile. No blocking action required.")
            recs.append("Continue standard telemetry collection and logging.")
            return recs

        # --- Fix #8: attack recommendations from specific class ---
        if "DOS" in pred or "DDOS" in pred or "HULK" in pred or "GOLDENEYE" in pred:
            recs.append("Trigger automated upstream rate-limiting and connection-rate caps on the targeted port.")
            recs.append("Enable IP reputation filtering and block top anomalous subnet emitters.")
            if "SLOWLORIS" in pred or "SLOWHTTP" in pred:
                recs.append("Deploy HTTP slow-connection timeout rules (not SYN-flood scrubbing — this is slow-HTTP).")
            else:
                recs.append("Activate SYN-cookie / SYN-flood scrubbing on the upstream border router.")
        elif "SCAN" in pred or "PORTSCAN" in pred:
            recs.append("Isolate scanner source IP at boundary firewall immediately.")
            recs.append("Block port sweeps; disable non-essential public-facing listening ports.")
            recs.append("Review firewall logs for associated lateral movement attempts.")
        elif "PATATOR" in pred or "BRUTE" in pred:
            recs.append("Enforce multi-factor authentication (MFA) and temporarily lock targeted accounts.")
            recs.append("Enable fail2ban / IP-drop rules after threshold authentication failures.")
            recs.append("Rotate credentials for any service targeted by this source IP.")
        elif "BOT" in pred or "INFILTRATION" in pred:
            recs.append("Immediate incident response: isolate affected endpoint from internal LAN.")
            recs.append("Inspect outbound DNS for C2 beaconing; sinkhole identified malicious domains.")
            recs.append("Preserve memory dump and logs for forensics before containment.")
        elif "HEARTBLEED" in pred:
            recs.append("Patch OpenSSL immediately (CVE-2014-0160). Rotate all TLS private keys and session tokens.")
            recs.append("Revoke and reissue any certificates exposed on the affected server.")
        elif "XSS" in pred or "SQL" in pred or "WEB" in pred:
            recs.append("Enable WAF rules for reflected XSS / SQL injection signatures.")
            recs.append("Review application logs for exfiltration or account takeover attempts.")
            recs.append("Apply input validation and parameterised query patches to the affected endpoint.")
        else:
            recs.append("Elevated risk detected. Quarantine suspect flow and notify SOC tier 2.")
            recs.append("Review full packet payload and firewall access logs.")
        return recs


# ---------------------------------------------------------------------------
# Severity constants  — Fix #12: keys match encoder.classes_ (underscore form)
# ---------------------------------------------------------------------------

ATTACK_BASE_SEVERITY = {
    "BENIGN":                       0.55,
    "FTP-Patator":                  0.55,
    "SSH-Patator":                  0.55,
    "DoS slowloris":                0.65,
    "DoS Slowhttptest":             0.65,
    "DoS GoldenEye":                0.70,
    "DoS Hulk":                     0.75,
    "Heartbleed":                   0.90,
    # Fix #12 — these keys must match LabelEncoder.classes_ (underscore form)
    "Web Attack_Brute Force":       0.60,
    "Web Attack_XSS":               0.65,
    "Web Attack_Sql Injection":     0.80,
    # Also accept en-dash form in case some datasets use it
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
        if len(self.rf_bundle) == 4:
            self._model, self._scaler, self._encoder, self._feature_columns = self.rf_bundle
            self._imputer_medians: dict[str, float] = {}
        else:
            self._model, self._scaler, self._encoder, self._feature_columns, self._imputer_medians = self.rf_bundle

        # K-Means — optional
        self._kmeans = None
        self._cluster_profiles: dict = {}
        try:
            from src.models.kmeans import load_kmeans, get_cluster_profiles
            self._kmeans = load_kmeans()
            self._cluster_profiles = get_cluster_profiles()
        except Exception:
            pass

        # Apriori rules — optional (explanation only — Fix #5)
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
        # 1. RF prediction (includes coverage metadata — Fix #1)
        rf_result = self._predict_rf(flow)
        coverage_meta = {
            "feature_coverage": rf_result.get("feature_coverage", 0.0),
            "supplied_features": rf_result.get("supplied_features", 0),
            "total_features": rf_result.get("total_features", len(self._feature_columns)),
            "missing_features": rf_result.get("missing_features", []),
        }

        # 2. Feature contributions — only supplied features (Fix #13)
        supplied_set = set(flow.keys()) & set(self._feature_columns)
        top_features = self._compute_feature_contributions(flow, supplied_set)

        # 3. K-Means cluster assignment
        cluster_id, cluster_profile_name, cluster_size, cluster_breakdown = (
            self._assign_cluster(flow)
        )

        # 4. Apriori pattern matching (explanation only — Fix #5/#7)
        matched_patterns = self._match_patterns(flow)

        # 5. Compute composite severity (Fix #3/#5)
        severity_score = self._compute_severity(
            rf_result, cluster_breakdown
        )
        severity = _severity_label(severity_score)

        # 6. Generate human-readable explanation
        explanation = self._generate_explanation(
            rf_result, severity, cluster_id, cluster_profile_name,
            matched_patterns, top_features, coverage_meta,
        )

        return ThreatAlert(
            prediction=rf_result["prediction"],
            confidence=rf_result["confidence"],
            probabilities=rf_result["probabilities"],
            feature_coverage=coverage_meta["feature_coverage"],
            supplied_features=coverage_meta["supplied_features"],
            total_features=coverage_meta["total_features"],
            missing_features=coverage_meta["missing_features"],
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
            if c.get("dominant_attack", "").upper() not in ("BENIGN", "UNKNOWN", "")
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

    def _compute_feature_contributions(
        self, flow: dict, supplied_set: set[str], top_n: int = 5
    ) -> list[dict]:
        """Estimate which *supplied* features contributed most to this prediction.

        Fix #13 — only features actually provided by the user are listed.
        Using feature importance * normalised deviation from training mean.
        """
        importances = self._model.feature_importances_
        scaler_means = self._scaler.mean_
        scaler_scales = self._scaler.scale_

        contributions = []
        for i, col in enumerate(self._feature_columns):
            # Fix #13 — skip features not in the submitted flow
            if col not in supplied_set:
                continue
            raw_val = float(flow.get(col, 0))
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
        """Assign the flow to a K-Means cluster and return profile info."""
        if self._kmeans is None:
            return -1, "Unknown", 0, {}

        df = pd.DataFrame([flow])
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = self._imputer_medians.get(col, 0.0)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[self._feature_columns]
        for col in self._feature_columns:
            bad = ~np.isfinite(df[col].values.astype(float))
            if bad.any():
                df.loc[df.index[bad], col] = self._imputer_medians.get(col, 0.0)

        scaled = self._scaler.transform(df)
        cluster_id = int(self._kmeans.predict(scaled)[0])

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
        self, flow: dict, max_matches: int = 5
    ) -> list[PatternMatch]:
        """Check which Apriori rules match the characteristics of this flow.

        Fix #7 — ATTACK= items are NEVER inserted at serve time.
        Fix #6 — item builder mirrors the training-time transaction builder.
        """
        if not self._apriori_rules:
            return []

        items = self._flow_to_items(flow)

        matched: list[PatternMatch] = []
        for rule in self._apriori_rules:
            ante = set(rule["antecedent"])
            # Fix #7 — skip rules containing ATTACK= in antecedent or consequent
            if any(item.startswith("ATTACK=") for item in ante):
                continue
            if any(item.startswith("ATTACK=") for item in rule.get("consequent", [])):
                continue
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

        matched.sort(key=lambda p: p.lift, reverse=True)
        return matched

    def _flow_to_items(self, flow: dict) -> set[str]:
        """Convert flow features to Apriori-style items for matching.

        Fix #6 — matches _flow_to_transaction in apriori.py exactly,
        including PAYLOAD=ZERO and WIN_SIZE items.
        Fix #7 — no ATTACK= item is ever inserted.
        """
        items: set[str] = set()

        # Destination port bucket
        from src.models.apriori import PORT_BUCKETS
        port = flow.get("Destination Port", -1)
        if port is not None and not (isinstance(port, float) and np.isnan(port)):
            try:
                port = int(port)
                matched_port = False
                for name, ports in PORT_BUCKETS.items():
                    if port in ports:
                        items.add("PORT={}".format(name))
                        matched_port = True
                        break
                if not matched_port and port > 0:
                    items.add("PORT=OTHER")
            except (TypeError, ValueError):
                pass

        # TCP flags
        for flag_col, label in [
            ("SYN Flag Count", "SYN_SET"),
            ("ACK Flag Count", "ACK_SET"),
            ("FIN Flag Count", "FIN_SET"),
            ("RST Flag Count", "RST_SET"),
        ]:
            val = flow.get(flag_col, 0)
            if val and val > 0:
                items.add(label)

        # Packet size category
        avg_size = flow.get("Average Packet Size", None)
        if avg_size is not None and not (isinstance(avg_size, float) and np.isnan(avg_size)):
            if avg_size < 100:
                items.add("PKT_SIZE=SMALL")
            elif avg_size < 1000:
                items.add("PKT_SIZE=MEDIUM")
            else:
                items.add("PKT_SIZE=LARGE")

        # Flow rate
        pps = flow.get("Flow Packets/s", None)
        if pps is not None and not (isinstance(pps, float) and np.isnan(pps)):
            if pps > 10000:
                items.add("RATE=HIGH_PPS")
            elif pps < 100:
                items.add("RATE=LOW_PPS")

        # Duration
        duration = flow.get("Flow Duration", None)
        if duration is not None and not (isinstance(duration, float) and np.isnan(duration)):
            if duration < 1000:
                items.add("DURATION=SHORT")
            elif duration < 100000:
                items.add("DURATION=MEDIUM")
            else:
                items.add("DURATION=LONG")

        # Fix #6 — zero-payload detection (present in training transactions)
        fwd_len = flow.get("Total Length of Fwd Packets", None)
        bwd_len = flow.get("Total Length of Bwd Packets", None)
        if fwd_len is not None and bwd_len is not None:
            try:
                if float(fwd_len) == 0.0 and float(bwd_len) == 0.0:
                    items.add("PAYLOAD=ZERO")
            except (TypeError, ValueError):
                pass

        # Fix #6 — window size category
        win = flow.get("Init_Win_bytes_forward", None)
        if win is not None:
            try:
                w = float(win)
                if w <= 0:
                    items.add("WIN_SIZE=ZERO")
                elif w < 256:
                    items.add("WIN_SIZE=SMALL")
            except (TypeError, ValueError):
                pass

        return items

    # ------------------------------------------------------------------
    # Internal: Severity computation
    # ------------------------------------------------------------------

    def _compute_severity(
        self,
        rf_result: dict,
        cluster_breakdown: dict,
    ) -> float:
        """Compute a composite severity score (0–1) from RF + K-Means.

        Fix #3 — RF signal = weighted sum of attack class probabilities
        (sum p_i * severity_i for i != BENIGN), replacing the broken
        base_sev * max(proba) formula that gave 0 whenever the top prediction
        was BENIGN — even when residual attack probability was non-zero.

        Fix #5 — Apriori term REMOVED from the numeric score to prevent
        benign DNS/small-packet traffic from accumulating +30 points.

        Weighting (revised):
            60%  RF weighted attack probability
            40%  Cluster attack dominance ratio (only when breakdown non-empty)
        """
        probabilities = rf_result.get("probabilities", {})

        # Fix #3 — weighted attack probability across ALL classes
        rf_signal = 0.0
        for label, p in probabilities.items():
            sev = ATTACK_BASE_SEVERITY.get(label, 0.5 if label.upper() != "BENIGN" else 0.0)
            rf_signal += p * sev
        # rf_signal is already in [0,1] since it's a convex combination

        # Fix #2 — cluster term only contributes when breakdown is real
        cluster_signal = 0.0
        if cluster_breakdown:
            attack_ratio = sum(
                v for k, v in cluster_breakdown.items()
                if k.upper() != "BENIGN"
            )
            cluster_signal = min(attack_ratio, 1.0)
            composite = 0.60 * rf_signal + 0.40 * cluster_signal
        else:
            # Cluster unavailable — rely entirely on RF
            composite = rf_signal

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
        coverage_meta: dict,
    ) -> str:
        """Generate a human-readable threat intelligence explanation."""
        prediction = rf_result["prediction"]
        confidence = rf_result["confidence"]
        coverage = coverage_meta.get("feature_coverage", 1.0)
        supplied = coverage_meta.get("supplied_features", 0)
        total = coverage_meta.get("total_features", 0)

        # Coverage warning
        coverage_note = ""
        if coverage < 1.0:
            coverage_note = (
                " [Coverage: {}/{} features ({:.0%}) — {} features zero-filled with training medians.]".format(
                    supplied, total, coverage, total - supplied
                )
            )

        if prediction.upper() == "BENIGN":
            return (
                "Traffic classified as BENIGN with {:.0%} confidence.{}".format(
                    confidence, coverage_note
                )
            )

        # Attack path
        # Compute attack probability mass
        probs = rf_result.get("probabilities", {})
        attack_prob = sum(p for k, p in probs.items() if k.upper() != "BENIGN")

        parts = [
            "\u26a0\ufe0f {} SEVERITY \u2014 {} detected with {:.0%} confidence (attack probability mass {:.0%}).".format(
                severity, prediction, confidence, attack_prob
            )
        ]

        if coverage_note:
            parts.append(coverage_note.strip(" []"))

        if cluster_id >= 0:
            if cluster_name and cluster_name != "Unknown":
                parts.append(
                    "Cluster #{} dominant behaviour: {}.".format(cluster_id, cluster_name)
                )
            else:
                parts.append(
                    "Cluster #{} (no label mapping yet — re-run K-Means with labels).".format(cluster_id)
                )

        if patterns:
            top_pattern = patterns[0]
            ante_str = " + ".join(top_pattern.antecedent)
            cons_str = ", ".join(top_pattern.consequent)
            parts.append(
                "Pattern match: [{}] \u2192 [{}] (lift: {:.2f}\u00d7) — explanatory only.".format(
                    ante_str, cons_str, top_pattern.lift
                )
            )

        if top_features:
            feat_names = [f["feature"] for f in top_features[:3]]
            parts.append("Key supplied features: {}.".format(", ".join(feat_names)))

        return " ".join(parts)


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
