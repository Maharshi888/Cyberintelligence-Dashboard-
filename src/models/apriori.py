"""
src/models/apriori.py
----------------------
Apriori association rule mining for discovering frequent attack combinations
in the CICIDS2017 dataset.

Requires: mlxtend  (pip install mlxtend)

Usage:
    from src.models.apriori import mine_patterns, get_top_rules
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RULES_PATH = PROCESSED_DIR / "apriori_rules.json"

LABEL_COL = "Label"

# Port categories used to build transactions
PORT_BUCKETS = {
    "HTTP": (80, 8080),
    "HTTPS": (443,),
    "DNS": (53,),
    "SSH": (22,),
    "FTP": (20, 21),
    "SMTP": (25, 587),
    "RDP": (3389,),
}


def _flow_to_transaction(row: pd.Series) -> list[str]:
    """Convert a single flow row into a list of discrete items for Apriori.

    Items are derived from:
    - Attack label (e.g. "ATTACK=DDoS")
    - Destination port bucket (e.g. "PORT=HTTP")
    - Flag combinations (e.g. "SYN_SET", "ACK_SET")
    - Packet size category (Small / Medium / Large)
    - Flow rate category (HIGH_PPS / LOW_PPS)
    - Duration category (SHORT / MEDIUM / LONG)
    - Payload indicator (ZERO_PAYLOAD)
    - Window size indicator (WIN_SIZE=SMALL / NORMAL)
    """
    items: list[str] = []

    # Attack label
    if LABEL_COL in row.index:
        label = str(row[LABEL_COL]).strip()
        items.append(f"ATTACK={label}")

    # Destination port bucket
    if "Destination Port" in row.index:
        port = int(row["Destination Port"]) if not pd.isna(row.get("Destination Port", np.nan)) else -1
        matched = False
        for name, ports in PORT_BUCKETS.items():
            if port in ports:
                items.append(f"PORT={name}")
                matched = True
                break
        if not matched and port > 0:
            items.append("PORT=OTHER")

    # TCP flags
    for flag_col, label in [
        ("SYN Flag Count", "SYN_SET"),
        ("ACK Flag Count", "ACK_SET"),
        ("FIN Flag Count", "FIN_SET"),
        ("RST Flag Count", "RST_SET"),
    ]:
        if flag_col in row.index and row.get(flag_col, 0) > 0:
            items.append(label)

    # Packet size category
    if "Average Packet Size" in row.index:
        size = row.get("Average Packet Size", 0)
        if pd.isna(size):
            size = 0
        if size < 100:
            items.append("PKT_SIZE=SMALL")
        elif size < 1000:
            items.append("PKT_SIZE=MEDIUM")
        else:
            items.append("PKT_SIZE=LARGE")

    # Flow rate category (packets per second)
    if "Flow Packets/s" in row.index:
        pps = row.get("Flow Packets/s", 0)
        if not pd.isna(pps):
            if pps > 10000:
                items.append("RATE=HIGH_PPS")
            elif pps < 100:
                items.append("RATE=LOW_PPS")

    # Duration category
    if "Flow Duration" in row.index:
        dur = row.get("Flow Duration", 0)
        if not pd.isna(dur):
            if dur < 1000:
                items.append("DURATION=SHORT")
            elif dur < 100000:
                items.append("DURATION=MEDIUM")
            else:
                items.append("DURATION=LONG")

    # Zero-payload detection (SYN floods, scanning)
    fwd_len = row.get("Total Length of Fwd Packets", 0) if "Total Length of Fwd Packets" in row.index else 0
    bwd_len = row.get("Total Length of Bwd Packets", 0) if "Total Length of Bwd Packets" in row.index else 0
    if not pd.isna(fwd_len) and not pd.isna(bwd_len):
        if fwd_len == 0 and bwd_len == 0:
            items.append("PAYLOAD=ZERO")

    # Initial window size indicator
    if "Init_Win_bytes_forward" in row.index:
        win = row.get("Init_Win_bytes_forward", 0)
        if not pd.isna(win):
            if win <= 0:
                items.append("WIN_SIZE=ZERO")
            elif win < 256:
                items.append("WIN_SIZE=SMALL")

    return list(set(items))  # deduplicate


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of flows to a boolean transaction matrix.

    Returns a DataFrame suitable for mlxtend's TransactionEncoder /
    apriori function (one-hot encoded items).
    """
    try:
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as e:
        raise ImportError("Install mlxtend: pip install mlxtend") from e

    transactions = df.apply(_flow_to_transaction, axis=1).tolist()
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)


def mine_patterns(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.6,
    max_rules: int = 100,
) -> list[dict]:
    """Mine association rules from a flows DataFrame.

    Args:
        df: Flows DataFrame (sampled subset recommended).
        min_support: Minimum itemset support threshold (0-1).
        min_confidence: Minimum rule confidence threshold (0-1).
        max_rules: Maximum number of rules to return.

    Returns:
        List of rule dicts with antecedent, consequent, support, confidence, lift.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as e:
        raise ImportError("Install mlxtend: pip install mlxtend") from e

    te_df = build_transactions(df)
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False).head(max_rules)

    result = []
    for _, row in rules.iterrows():
        result.append(
            {
                "antecedent": sorted(list(row["antecedents"])),
                "consequent": sorted(list(row["consequents"])),
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
            }
        )

    _save_rules(result)
    return result


def _save_rules(rules: list[dict]) -> None:
    """Persist mined rules to processed/apriori_rules.json."""
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


def get_top_rules(n: int = 20) -> list[dict]:
    """Load the top-N cached rules sorted by lift.

    Returns an empty list if mining hasn't been run yet.
    """
    if not RULES_PATH.exists():
        return []
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)
    # Sort by lift descending in case file is unsorted
    rules.sort(key=lambda r: r.get("lift", 0), reverse=True)
    return rules[:n]
