/**
 * frontend/js/api.js
 * ------------------
 * API client — all fetch calls to the FastAPI backend.
 * Falls back to built-in demo data if the server is unreachable.
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : '';

// ── Demo / fallback data ─────────────────────────────────────────────────────

const DEMO_STATS = {
  total_flows: 2830743,
  attack_flows: 557647,
  benign_flows: 2273096,
  attack_types: 14,
  label_distribution: {
    "BENIGN": 2273096,
    "DoS Hulk": 231073,
    "PortScan": 158930,
    "DDoS": 128027,
    "DoS GoldenEye": 10293,
    "FTP-Patator": 7938,
    "SSH-Patator": 5897,
    "DoS slowloris": 5796,
    "DoS Slowhttptest": 5499,
    "Bot": 1966,
    "Web Attack – Brute Force": 1507,
    "Web Attack – XSS": 652,
    "Infiltration": 36,
    "Web Attack – Sql Injection": 21,
    "Heartbleed": 11
  }
};

const DEMO_TRENDS = {
  hours: ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00",
          "08:00","09:00","10:00","11:00","12:00","13:00","14:00","15:00",
          "16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"],
  series: {
    "DoS Hulk":     [420,310,280,250,290,380,650,9800,11500,10200,9800,8700,8900,9200,10100,9600,8400,7200,6000,4200,3100,2200,1400,820],
    "PortScan":     [180,130,100,90,110,160,350,5200,6100,5400,5200,4600,4700,4900,5400,5100,4500,3800,3200,2200,1600,1100,700,380],
    "DDoS":         [310,230,200,170,200,270,480,7100,8400,7400,7100,6300,6500,6700,7400,7000,6100,5200,4300,3000,2200,1600,1000,590],
    "Bot":          [30,22,18,16,20,28,55,780,920,810,780,690,710,730,800,760,660,560,470,320,240,170,110,65],
    "FTP-Patator":  [60,45,38,34,40,55,100,1400,1650,1460,1400,1250,1280,1320,1450,1380,1200,1020,850,590,430,310,200,120]
  }
};

const DEMO_MODEL_REPORT = {
  accuracy: 0.9973,
  label_count: 15,
  classes: ["BENIGN","Bot","DDoS","DoS GoldenEye","DoS Hulk","DoS Slowhttptest",
            "DoS slowloris","FTP-Patator","Heartbleed","Infiltration","PortScan",
            "SSH-Patator","Web Attack – Brute Force","Web Attack – Sql Injection","Web Attack – XSS"],
  classification_report: {
    "weighted avg": { "precision": 0.9974, "recall": 0.9973, "f1-score": 0.9973 }
  }
};

const DEMO_FEATURES = {
  features: [
    { feature: "Flow Duration",              importance: 0.1423 },
    { feature: "Fwd Packet Length Max",      importance: 0.0921 },
    { feature: "Bwd Packet Length Max",      importance: 0.0817 },
    { feature: "Flow Bytes/s",               importance: 0.0742 },
    { feature: "Flow IAT Max",               importance: 0.0698 },
    { feature: "Fwd IAT Max",                importance: 0.0632 },
    { feature: "Packet Length Variance",     importance: 0.0589 },
    { feature: "Avg Bwd Segment Size",       importance: 0.0521 },
    { feature: "Average Packet Size",        importance: 0.0498 },
    { feature: "Init_Win_bytes_forward",     importance: 0.0453 },
    { feature: "Destination Port",           importance: 0.0412 },
    { feature: "Total Backward Packets",     importance: 0.0387 },
    { feature: "Bwd Packet Length Mean",     importance: 0.0341 },
    { feature: "min_seg_size_forward",       importance: 0.0312 },
    { feature: "Packet Length Std",          importance: 0.0287 },
    { feature: "Bwd IAT Total",              importance: 0.0253 },
    { feature: "Fwd Header Length",          importance: 0.0231 },
    { feature: "Flow IAT Std",               importance: 0.0198 },
    { feature: "ACK Flag Count",             importance: 0.0187 },
    { feature: "SYN Flag Count",             importance: 0.0176 }
  ]
};

// ── Fetch helper ─────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`[API] ${path} failed (${err.message}) — using demo data`);
    return null;
  }
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function fetchStats() {
  return (await apiFetch('/api/stats')) ?? DEMO_STATS;
}

export async function fetchTrends() {
  return (await apiFetch('/api/trends')) ?? DEMO_TRENDS;
}

export async function fetchModelReport() {
  return (await apiFetch('/api/model-report')) ?? DEMO_MODEL_REPORT;
}

export async function fetchFeatures(topN = 20) {
  return (await apiFetch(`/api/features?top_n=${topN}`)) ?? DEMO_FEATURES;
}

export async function fetchForensics() {
  return (await apiFetch('/api/forensics')) ?? null;
}

export async function fetchClusters() {
  return (await apiFetch('/api/clusters')) ?? { status: 'not_computed', clusters: [] };
}

export async function fetchClusterStatus() {
  return (await apiFetch('/api/clusters/status')) ?? { status: 'idle', message: '' };
}

export async function triggerClustering(k = 8, sampleSize = 50000) {
  return apiFetch(`/api/clusters/train?k=${k}&sample_size=${sampleSize}`, { method: 'POST' });
}

export async function fetchPatterns(topN = 20) {
  return (await apiFetch(`/api/patterns?top_n=${topN}`)) ?? { status: 'not_computed', rules: [] };
}

export async function fetchPatternStatus() {
  return (await apiFetch('/api/patterns/status')) ?? { status: 'idle', message: '' };
}

export async function triggerMining(sampleSize = 30000, minSupport = 0.05, minConf = 0.6) {
  return apiFetch(
    `/api/patterns/mine?sample_size=${sampleSize}&min_support=${minSupport}&min_confidence=${minConf}`,
    { method: 'POST' }
  );
}

export async function predictFlow(features) {
  const res = await apiFetch('/api/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
  if (!res) {
    // Demo prediction
    const labels = Object.keys(DEMO_STATS.label_distribution);
    const label = labels[Math.floor(Math.random() * labels.length)];
    const conf = 0.72 + Math.random() * 0.27;
    return {
      prediction: label,
      confidence: conf,
      probabilities: Object.fromEntries(labels.map(l => [l, l === label ? conf : Math.random() * 0.05]))
    };
  }
  return res;
}

export async function fetchLiveFeed(limit = 20) {
  return (await apiFetch(`/api/live-feed?limit=${limit}`)) ?? { feed: [] };
}

// ── Threat Intelligence ──────────────────────────────────────────────────────

export async function assessThreatIntel(features) {
  const res = await apiFetch('/api/threat-intel/assess', {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
  if (!res) {
    // Demo threat assessment
    const labels = Object.keys(DEMO_STATS.label_distribution);
    const label = labels[Math.floor(Math.random() * labels.length)];
    const conf = 0.72 + Math.random() * 0.27;
    const isBenign = label.toUpperCase() === 'BENIGN';
    return {
      prediction: label,
      confidence: conf,
      severity: isBenign ? 'LOW' : ['MEDIUM', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 3)],
      severity_score: isBenign ? 0.05 : 0.3 + Math.random() * 0.6,
      cluster_id: Math.floor(Math.random() * 8),
      cluster_profile_name: label,
      cluster_size: 1000 + Math.floor(Math.random() * 50000),
      cluster_label_breakdown: { [label]: 0.72, "BENIGN": 0.18, "Other": 0.10 },
      matched_patterns: isBenign ? [] : [
        { antecedent: ["SYN_SET", "RATE=HIGH_PPS"], consequent: [`ATTACK=${label}`],
          support: 0.08, confidence: 0.85, lift: 3.2 + Math.random() * 3 }
      ],
      top_contributing_features: [
        { feature: "Destination Port", importance: 0.14, value: 80, contribution_score: 0.42 },
        { feature: "Flow Packets/s", importance: 0.09, value: 18000, contribution_score: 0.38 },
        { feature: "Flow Duration", importance: 0.12, value: 109, contribution_score: 0.28 },
      ],
      explanation: isBenign
        ? `Traffic classified as BENIGN with ${(conf*100).toFixed(0)}% confidence. No threat indicators detected.`
        : `⚠️ HIGH SEVERITY — ${label} detected with ${(conf*100).toFixed(0)}% confidence. Flow assigned to Cluster #${Math.floor(Math.random()*8)}. Matches attack pattern.`,
      timestamp: new Date().toISOString(),
    };
  }
  return res;
}

export async function fetchThreatSummary() {
  return (await apiFetch('/api/threat-intel/summary')) ?? {
    model_status: { random_forest: false, kmeans: false, apriori: false },
    attack_classes: Object.keys(DEMO_STATS.label_distribution),
    num_clusters: 0,
    attack_clusters: 0,
    num_apriori_rules: 0,
    silhouette_score: null,
    top_attack_patterns: [],
  };
}

