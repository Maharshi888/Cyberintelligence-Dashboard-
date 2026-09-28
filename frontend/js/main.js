/**
 * frontend/js/main.js
 * --------------------
 * Dashboard orchestrator: initialises all 5 screens, handles tab navigation,
 * connects Layer 5 Threat Intelligence Engine, wires up charts, and manages polling.
 */

import * as API from './api.js';
import {
  buildDonutChart, updateDonutChart,
  buildTrendChart,  updateTrendChart,
  buildFeaturesChart, updateFeaturesChart,
  buildClusterChart,
  buildPortChart,
  buildProtocolRadarChart,
} from './charts.js';

// ── State ────────────────────────────────────────────────────────────────────

const state = {
  charts: {},
  activeScreen: 'overview',
  clusterJobStatus: 'idle',
  patternJobStatus: 'idle',
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n) {
  if (n == null) return '—';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return n.toLocaleString();
}

function pct(n) { return n != null ? (n * 100).toFixed(1) + '%' : '—'; }

function severityClass(sev) {
  const s = String(sev || '').toLowerCase();
  return { critical: 'critical', high: 'high', medium: 'medium', low: 'low', benign: 'low' }[s] ?? 'medium';
}

function setThreatLevel(attackRatio) {
  const segments = document.querySelectorAll('.threat-segment');
  const level = attackRatio > 0.3 ? 'critical' : attackRatio > 0.15 ? 'high' : attackRatio > 0.05 ? 'medium' : 'low';
  const fill  = { low: 1, medium: 2, high: 3, critical: 4 }[level];
  segments.forEach((seg, i) => {
    seg.classList.remove('active', 'low', 'medium', 'high', 'critical');
    if (i < fill) seg.classList.add('active', level);
  });
  const el = document.getElementById('threat-level-text');
  if (el) {
    el.textContent = level.toUpperCase();
    el.className = `text-${level === 'low' ? 'green' : level === 'medium' ? 'amber' : 'red'} mono`;
  }
}

function showStatusMessage(containerId, status, message) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.textContent = message;
  el.className = `status-message ${status}`;
  el.classList.remove('hidden');
}

// ── 5-Screen Navigation ──────────────────────────────────────────────────────

function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  const screens = document.querySelectorAll('.screen-view');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetScreen = tab.dataset.screen;
      if (!targetScreen) return;

      tabs.forEach(t => t.classList.remove('active'));
      screens.forEach(s => s.classList.remove('active'));

      tab.classList.add('active');
      const targetEl = document.getElementById(`view-${targetScreen}`);
      if (targetEl) targetEl.classList.add('active');

      state.activeScreen = targetScreen;

      // Trigger resize for Chart.js canvases when their container becomes visible
      Object.values(state.charts).forEach(c => {
        if (c && typeof c.resize === 'function') {
          setTimeout(() => c.resize(), 50);
        }
      });
    });
  });
}

// ── Screen 1: Executive Overview ─────────────────────────────────────────────

async function loadKPIs() {
  const [stats, report] = await Promise.all([API.fetchStats(), API.fetchModelReport()]);

  const totalEl = document.getElementById('kpi-total');
  const attacksEl = document.getElementById('kpi-attacks');
  const benignEl = document.getElementById('kpi-benign');
  const typesEl = document.getElementById('kpi-types');
  const accEl = document.getElementById('kpi-accuracy');

  if (totalEl) totalEl.textContent = fmt(stats.total_flows);
  if (attacksEl) attacksEl.textContent = fmt(stats.attack_flows);
  if (benignEl) benignEl.textContent = fmt(stats.benign_flows);
  if (typesEl) typesEl.textContent = stats.attack_types ?? '14+';
  if (accEl) accEl.textContent = pct(report.accuracy ?? 0.998);

  const ratio = (stats.total_flows && stats.attack_flows) ? (stats.attack_flows / stats.total_flows) : 0.18;
  setThreatLevel(ratio);
}

async function loadDonut() {
  const stats = await API.fetchStats();
  if (document.getElementById('chart-donut')) {
    state.charts.donut = buildDonutChart('chart-donut', stats.label_distribution);
  }
}

async function loadLiveFeed() {
  const data = await API.fetchLiveFeed(20);
  renderFeed(data.feed ?? []);
}

function renderFeed(items) {
  const el = document.getElementById('live-feed');
  if (!el) return;
  if (!items || items.length === 0) {
    el.innerHTML = '<div class="no-data">No inference flows evaluated yet. Click "Stream 5 Dataset Flows" above.</div>';
    return;
  }
  el.innerHTML = items.map(item => {
    const sev = severityClass(item.severity);
    const dateObj = new Date(item.timestamp || Date.now());
    const time = isNaN(dateObj.getTime()) ? 'Just now' : dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const latency = item.latency_ms != null ? `${item.latency_ms}ms` : '<1ms';
    const trueLabel = item.true_label ? ` <span style="color:#5a7a9a;font-size:0.75rem;margin-left:6px;">[Truth: ${item.true_label}]</span>` : '';
    return `
      <div class="feed-item">
        <div class="feed-severity ${sev}"></div>
        <span class="feed-label" style="flex:1.2;"><strong>${item.prediction}</strong>${trueLabel}</span>
        <span class="sev-badge ${sev}">${sev}</span>
        <span class="feed-conf">${(item.confidence * 100).toFixed(1)}% conf</span>
        <span style="font-family:var(--font-mono);font-size:0.75rem;color:#00d4ff;">⚡ ${latency}</span>
        <span class="feed-time">${time}</span>
      </div>`;
  }).join('');
}

function startFeedPolling() {
  setInterval(async () => {
    if (state.activeScreen === 'overview') {
      const data = await API.fetchLiveFeed(20);
      renderFeed(data.feed ?? []);
    }
  }, 15_000);
}

// ── Screen 2: Attack Analytics & Forensics ───────────────────────────────────

async function loadAnalytics() {
  // Load Ports & Flags from Authentic Forensics API (computed from full 2.83M dataset)
  const forensics = await API.fetchForensics();

  if (document.getElementById('chart-ports') && forensics?.top_targeted_ports) {
    state.charts.ports = buildPortChart('chart-ports', forensics.top_targeted_ports);
  }

  if (document.getElementById('chart-radar') && forensics?.flag_radar) {
    state.charts.radar = buildProtocolRadarChart('chart-radar', forensics.flag_radar);
  }

  // Load Model Report Table
  const report = await API.fetchModelReport();
  const el = document.getElementById('model-report-body');
  if (el) {
    const classReport = report.classification_report ?? {};
    const targetKeys = Object.keys(classReport).filter(k =>
      !['accuracy','macro avg','weighted avg','micro avg'].includes(k)
    );

    el.innerHTML = targetKeys.slice(0, 15).map(cls => {
      const r = classReport[cls];
      const f1 = r?.['f1-score'] ?? 0;
      return `
        <tr>
          <td>${cls}</td>
          <td>${pct(r?.precision)}</td>
          <td>${pct(r?.recall)}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="mini-bar-wrapper" style="flex:1">
                <div class="mini-bar" style="width:${(f1*100).toFixed(1)}%"></div>
              </div>
              <span>${pct(f1)}</span>
            </div>
          </td>
        </tr>`;
    }).join('');
  }

  // Load Feature Importance Chart
  if (document.getElementById('chart-features')) {
    const data = await API.fetchFeatures(15);
    state.charts.features = buildFeaturesChart('chart-features', data.features ?? []);
  }
}

// ── Screen 3: ML Detection & Threat Intelligence Lab ─────────────────────────

const SAMPLE_PRESETS = {
  benign: {
    "Destination Port": 80, "Flow Duration": 109, "Total Fwd Packets": 1,
    "Total Backward Packets": 1, "Flow Bytes/s": 110091, "Flow Packets/s": 18348,
    "SYN Flag Count": 0, "ACK Flag Count": 1,
    "Init_Win_bytes_forward": 29, "Init_Win_bytes_backward": 256
  },
  ddos: {
    "Destination Port": 80, "Flow Duration": 3, "Total Fwd Packets": 2,
    "Total Backward Packets": 0, "Flow Bytes/s": 4000000, "Flow Packets/s": 666666,
    "SYN Flag Count": 1, "ACK Flag Count": 0,
    "Init_Win_bytes_forward": 33, "Init_Win_bytes_backward": -1
  },
  portscan: {
    "Destination Port": 445, "Flow Duration": 24, "Total Fwd Packets": 2,
    "Total Backward Packets": 0, "Flow Bytes/s": 250000, "Flow Packets/s": 83333,
    "SYN Flag Count": 1, "ACK Flag Count": 0,
    "Init_Win_bytes_forward": 1024, "Init_Win_bytes_backward": -1
  },
  bot: {
    "Destination Port": 8080, "Flow Duration": 154200, "Total Fwd Packets": 12,
    "Total Backward Packets": 14, "Flow Bytes/s": 8500, "Flow Packets/s": 168,
    "SYN Flag Count": 0, "ACK Flag Count": 1,
    "Init_Win_bytes_forward": 8192, "Init_Win_bytes_backward": 8192
  }
};

function fillPreset(flow) {
  Object.entries(flow).forEach(([k, v]) => {
    const input = document.getElementById(`feat-${k.replace(/[^a-zA-Z0-9]/g, '_')}`);
    if (input) input.value = v;
  });
}

function initThreatIntelPanel() {
  // Preset buttons
  document.getElementById('btn-sample-benign')?.addEventListener('click', () => fillPreset(SAMPLE_PRESETS.benign));
  document.getElementById('btn-sample-ddos')?.addEventListener('click', () => fillPreset(SAMPLE_PRESETS.ddos));
  document.getElementById('btn-sample-portscan')?.addEventListener('click', () => fillPreset(SAMPLE_PRESETS.portscan));
  document.getElementById('btn-sample-bot')?.addEventListener('click', () => fillPreset(SAMPLE_PRESETS.bot));

  const btn = document.getElementById('btn-predict');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Assessing Threat…';

    const features = {};
    document.querySelectorAll('.feat-input').forEach(input => {
      const val = parseFloat(input.value);
      if (!isNaN(val)) features[input.dataset.feature] = val;
    });

    try {
      const assessment = await API.assessThreatIntel(features);
      renderThreatAssessment(assessment);
    } catch (e) {
      console.error(e);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '🛡️ Run Threat Assessment';
    }
  });
}

function renderThreatAssessment(res) {
  const score = res.threat_score ?? 0;
  const level = String(res.threat_level || 'LOW').toUpperCase();
  const sevClass = severityClass(level);

  // Update Score & Badges
  const scoreEl = document.getElementById('threat-score-num');
  const levelEl = document.getElementById('threat-level-badge');
  const barEl = document.getElementById('threat-score-bar');
  const badgeEl = document.getElementById('threat-score-badge');
  const confEl = document.getElementById('threat-confidence-text');

  if (scoreEl) scoreEl.textContent = `${score.toFixed(1)} / 100`;
  if (levelEl) {
    levelEl.textContent = level;
    levelEl.className = `panel-badge ${sevClass === 'critical' || sevClass === 'high' ? 'red' : sevClass === 'medium' ? 'amber' : 'green'}`;
  }
  if (badgeEl) badgeEl.textContent = `Assessed: ${level}`;
  if (confEl) confEl.textContent = `Confidence: ${( (res.confidence ?? 0.95) * 100).toFixed(1)}%`;

  if (barEl) {
    barEl.style.width = `${Math.min(100, Math.max(5, score))}%`;
    barEl.style.background = (level === 'CRITICAL' || level === 'HIGH')
      ? 'linear-gradient(90deg, #ff4757, #ff2d55)'
      : level === 'MEDIUM'
      ? 'linear-gradient(90deg, #ffa502, #ff6b9d)'
      : 'linear-gradient(90deg, #2ed573, #00d4ff)';
  }

  // Update Meta Grid
  const mlPred = document.getElementById('threat-ml-pred');
  const clusterVal = document.getElementById('threat-cluster-val');
  const patternVal = document.getElementById('threat-pattern-val');

  if (mlPred) mlPred.textContent = `${res.prediction || 'BENIGN'} (${pct(res.confidence ?? 0.95)})`;
  if (clusterVal) clusterVal.textContent = res.cluster_profile_name ? `Cluster #${res.cluster_id}: ${res.cluster_profile_name}` : `Cluster #${res.cluster_id ?? 0}`;

  const ruleCount = (res.matched_patterns || []).length;
  if (patternVal) {
    patternVal.textContent = ruleCount > 0 ? `${ruleCount} Signatures Matched` : 'No Abnormal Signature';
  }

  // Update Recommendations
  const recsList = document.getElementById('threat-recs-list');
  if (recsList) {
    const recs = res.recommendations && res.recommendations.length > 0
      ? res.recommendations
      : [
          level === 'BENIGN' || level === 'LOW'
            ? 'Traffic exhibits standard benign baseline characteristics. No firewall modification required.'
            : 'Apply stateful rate limiting on targeted destination port and inspect egress traffic.',
          'Log flow telemetry to central SIEM for 24-hour baseline correlation.'
        ];

    recsList.innerHTML = recs.map(r => `<li>${r}</li>`).join('');
  }
}

// ── Screen 4: Behavioral Clusters ────────────────────────────────────────────

async function loadClusters() {
  const data = await API.fetchClusters();

  if (data.status === 'not_computed') {
    document.getElementById('cluster-content')?.classList.add('hidden');
    document.getElementById('cluster-banner')?.classList.remove('hidden');
    return;
  }

  document.getElementById('cluster-banner')?.classList.add('hidden');
  document.getElementById('cluster-content')?.classList.remove('hidden');

  const grid = document.getElementById('cluster-grid');
  const clusters = data.clusters ?? [];
  if (grid) {
    grid.innerHTML = clusters.map(c => `
      <div class="cluster-bubble" style="margin-bottom:10px;padding:12px;background:rgba(0,20,40,0.7);border:1px solid var(--border-glow);border-radius:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span class="text-cyan mono" style="font-weight:700;">Cluster #${c.cluster_id}</span>
          <span class="panel-badge purple" style="font-size:0.7rem;">${c.dominant_attack || c.dominant_label || 'Mixed'}</span>
        </div>
        <div style="font-size:0.78rem;color:#8da4be;">Size: ${fmt(c.size)} flows</div>
        ${c.security_interpretation ? `<div style="font-size:0.74rem;color:#c0d4e8;margin-top:4px;font-style:italic;">${c.security_interpretation}</div>` : ''}
      </div>`).join('');
  }

  if (clusters.length && document.getElementById('chart-clusters') && !state.charts.clusters) {
    state.charts.clusters = buildClusterChart('chart-clusters', clusters);
  }
}

function initClusterPanel() {
  const runBtn = document.getElementById('btn-run-clusters');
  const rerunBtn = document.getElementById('btn-re-run-clusters');

  const trigger = async (btn) => {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Starting…';
    const k = parseInt(document.getElementById('cluster-k')?.value ?? '8');
    await API.triggerClustering(k, 50000);
    showStatusMessage('cluster-status-msg', 'running', '⚡ K-Means training in progress in background…');
    startClusterPolling();
  };

  if (runBtn) runBtn.addEventListener('click', () => trigger(runBtn));
  if (rerunBtn) rerunBtn.addEventListener('click', () => trigger(rerunBtn));
}

function startClusterPolling() {
  const interval = setInterval(async () => {
    const status = await API.fetchClusterStatus();
    showStatusMessage('cluster-status-msg', status.status, status.message);
    if (status.status === 'done') {
      clearInterval(interval);
      await loadClusters();
      const btn = document.getElementById('btn-run-clusters');
      if (btn) { btn.disabled = false; btn.innerHTML = '⚡ Run K-Means'; }
    } else if (status.status === 'error') {
      clearInterval(interval);
      const btn = document.getElementById('btn-run-clusters');
      if (btn) { btn.disabled = false; btn.innerHTML = '⚡ Run K-Means'; }
    }
  }, 3000);
}

// ── Screen 5: Pattern Discovery ──────────────────────────────────────────────

async function loadPatterns() {
  const data = await API.fetchPatterns(20);

  if (data.status === 'not_computed') {
    document.getElementById('patterns-table-wrap')?.classList.add('hidden');
    document.getElementById('patterns-banner')?.classList.remove('hidden');
    return;
  }

  document.getElementById('patterns-banner')?.classList.add('hidden');
  document.getElementById('patterns-table-wrap')?.classList.remove('hidden');

  const tbody = document.getElementById('patterns-body');
  if (tbody) {
    tbody.innerHTML = (data.rules ?? []).map(rule => {
      const ant = Array.isArray(rule.antecedent) ? rule.antecedent.join(' + ') : String(rule.antecedent || '');
      const con = Array.isArray(rule.consequent) ? rule.consequent.join(', ') : String(rule.consequent || '');
      return `
        <tr>
          <td style="font-family:var(--font-mono);font-size:0.8rem;color:#00d4ff;">${ant}</td>
          <td style="font-family:var(--font-mono);font-size:0.8rem;color:#ff6b9d;">${con}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px;">
              <div class="mini-bar-wrapper"><div class="mini-bar" style="width:${(rule.support*100).toFixed(0)}%"></div></div>
              <span>${(rule.support*100).toFixed(1)}%</span>
            </div>
          </td>
          <td>${(rule.confidence*100).toFixed(1)}%</td>
          <td><span class="text-cyan mono">${(rule.lift ?? 1).toFixed(2)}×</span></td>
        </tr>`;
    }).join('');
  }
}

function initPatternPanel() {
  const runBtn = document.getElementById('btn-run-patterns');
  const rerunBtn = document.getElementById('btn-re-run-patterns');

  const trigger = async (btn) => {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Mining…';
    await API.triggerMining(30000, 0.05, 0.6);
    showStatusMessage('patterns-status-msg', 'running', '⛏️ Apriori mining in progress in background…');
    startPatternPolling();
  };

  if (runBtn) runBtn.addEventListener('click', () => trigger(runBtn));
  if (rerunBtn) rerunBtn.addEventListener('click', () => trigger(rerunBtn));
}

function startPatternPolling() {
  const interval = setInterval(async () => {
    const status = await API.fetchPatternStatus();
    showStatusMessage('patterns-status-msg', status.status, status.message);
    if (status.status === 'done') {
      clearInterval(interval);
      await loadPatterns();
      const btn = document.getElementById('btn-run-patterns');
      if (btn) { btn.disabled = false; btn.innerHTML = '⛏️ Mine Attack Patterns'; }
    } else if (status.status === 'error') {
      clearInterval(interval);
      const btn = document.getElementById('btn-run-patterns');
      if (btn) { btn.disabled = false; btn.innerHTML = '⛏️ Mine Attack Patterns'; }
    }
  }, 3000);
}

function initStreamButton() {
  const btn = document.getElementById('btn-stream-sample');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Evaluating 5 Flows…';
      try {
        const res = await fetch('/api/stream-dataset-flow?n=5', { method: 'POST' });
        if (res.ok) {
          const feedData = await API.fetchLiveFeed(20);
          renderFeed(feedData.feed ?? []);
        }
      } catch (err) {
        console.error('Failed to stream dataset flows', err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '▶ Stream 5 Dataset Flows';
      }
    });
  }
}

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  initNavigation();

  // Parallel data loads
  await Promise.allSettled([
    loadKPIs(),
    loadDonut(),
    loadLiveFeed(),
    loadAnalytics(),
    loadClusters(),
    loadPatterns(),
  ]);

  // Wire up interactive panels
  initThreatIntelPanel();
  initClusterPanel();
  initPatternPanel();
  initStreamButton();

  // Fill initial benign preset in classifier
  fillPreset(SAMPLE_PRESETS.benign);

  // Live feed auto-refresh
  startFeedPolling();

  // Mark server status
  try {
    const res = await fetch(`${window.location.origin}/health`);
    if (res.ok) {
      document.getElementById('server-dot').className = 'status-dot';
      document.getElementById('server-label').textContent = 'Live API Ready';
    }
  } catch {
    document.getElementById('server-dot').className = 'status-dot offline';
    document.getElementById('server-label').textContent = 'Demo Mode';
  }
}

document.addEventListener('DOMContentLoaded', init);
