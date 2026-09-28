"""
src/data/compute_forensics.py
-----------------------------
Computes authentic forensic statistics directly from the 2.83M raw dataset (combinenew.csv):
1. Top targeted destination ports (with common service mappings)
2. Attack vs Benign TCP Flag profiles (SYN, ACK, FIN, RST, PSH, URG, ECE)
3. Protocol and traffic behavioral metrics

Saves output to data/processed/forensics.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "raw" / "combinenew.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PORT_NAMES = {
    80: "HTTP",
    443: "HTTPS",
    22: "SSH",
    21: "FTP",
    53: "DNS",
    8080: "HTTP-Alt",
    445: "SMB",
    139: "NetBIOS",
    25: "SMTP",
    110: "POP3",
    143: "IMAP",
    3389: "RDP",
    23: "Telnet",
    67: "DHCP",
    68: "DHCP",
    123: "NTP",
    161: "SNMP",
    389: "LDAP",
}

def compute_forensics():
    print(f"Reading dataset in chunks from {RAW_CSV}...")
    
    port_counts = {}
    attack_port_counts = {}
    
    flag_cols = [
        "SYN Flag Count",
        "ACK Flag Count",
        "FIN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "URG Flag Count",
        "ECE Flag Count",
    ]
    
    attack_flag_sums = {col: 0.0 for col in flag_cols}
    benign_flag_sums = {col: 0.0 for col in flag_cols}
    total_attack_rows = 0
    total_benign_rows = 0

    chunksize = 250_000
    for i, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=chunksize, low_memory=False)):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        
        # Clean destination port
        chunk["Destination Port"] = pd.to_numeric(chunk["Destination Port"], errors="coerce").fillna(0).astype(int)
        
        # Clean flag cols
        for col in flag_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
        
        is_attack = chunk["Label"].astype(str).str.strip().str.upper() != "BENIGN"
        attack_chunk = chunk[is_attack]
        benign_chunk = chunk[~is_attack]
        
        # Ports
        for port, cnt in chunk["Destination Port"].value_counts().head(20).items():
            port_int = int(port)
            if port_int > 0:
                port_counts[port_int] = port_counts.get(port_int, 0) + int(cnt)
            
        for port, cnt in attack_chunk["Destination Port"].value_counts().head(20).items():
            port_int = int(port)
            if port_int > 0:
                attack_port_counts[port_int] = attack_port_counts.get(port_int, 0) + int(cnt)
            
        # Flags
        for col in flag_cols:
            if col in chunk.columns:
                attack_flag_sums[col] += float(attack_chunk[col].sum())
                benign_flag_sums[col] += float(benign_chunk[col].sum())
                
        total_attack_rows += len(attack_chunk)
        total_benign_rows += len(benign_chunk)
        print(f"Processed chunk {i+1} ({len(chunk):,} rows)...")

    # Format top ports (attacks)
    sorted_ports = sorted(attack_port_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    ports_data = {}
    for port, count in sorted_ports:
        service = PORT_NAMES.get(port, "")
        label = f"Port {port} ({service})" if service else f"Port {port}"
        ports_data[label] = count

    # Format radar flags (normalize as percentage of flows having the flag)
    flag_labels = ["SYN Flag", "ACK Flag", "FIN Flag", "RST Flag", "PSH Flag", "URG Flag", "ECE Flag"]
    attack_percentages = []
    benign_percentages = []
    
    for col in flag_cols:
        atk_pct = round((attack_flag_sums[col] / max(1, total_attack_rows)) * 100, 2)
        ben_pct = round((benign_flag_sums[col] / max(1, total_benign_rows)) * 100, 2)
        attack_percentages.append(atk_pct)
        benign_percentages.append(ben_pct)

    forensics = {
        "dataset_total_flows": total_attack_rows + total_benign_rows,
        "attack_flows": total_attack_rows,
        "benign_flows": total_benign_rows,
        "top_targeted_ports": ports_data,
        "flag_radar": {
            "labels": flag_labels,
            "attack": attack_percentages,
            "benign": benign_percentages,
        }
    }

    out_path = PROCESSED_DIR / "forensics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forensics, f, indent=2)
        
    print(f"Forensics successfully computed and saved to {out_path}!")

if __name__ == "__main__":
    compute_forensics()
