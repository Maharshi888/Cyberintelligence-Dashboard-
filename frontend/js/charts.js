/**
 * frontend/js/charts.js
 * ----------------------
 * Chart.js initialisation and update helpers for all dashboard panels.
 * Each function returns the Chart instance so callers can update data later.
 */

// ── Chart.js global defaults ─────────────────────────────────────────────────

Chart.defaults.color = '#5a7a9a';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(4, 10, 20, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(0, 212, 255, 0.3)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.titleColor = '#00d4ff';
Chart.defaults.plugins.tooltip.bodyColor = '#a8c4e0';

const PALETTE = [
  '#00d4ff', '#a855f7', '#ff4757', '#ffa502', '#2ed573',
  '#ff6b9d', '#00b4d8', '#7b2d8b', '#f7971e', '#43e97b',
  '#667eea', '#f093fb', '#4facfe', '#ff6e7f', '#b8c6db'
];

// ── Attack Distribution Donut ────────────────────────────────────────────────

export function buildDonutChart(canvasId, distribution) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const labels = Object.keys(distribution);
  const data = Object.values(distribution);

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: labels.map((_, i) => PALETTE[i % PALETTE.length] + 'cc'),
        borderColor:     labels.map((_, i) => PALETTE[i % PALETTE.length]),
        borderWidth: 1.5,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { position: 'right', labels: { font: { size: 10 }, padding: 10 } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = ((ctx.parsed / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
            }
          }
        }
      },
      animation: { animateRotate: true, duration: 900 }
    }
  });
}

export function updateDonutChart(chart, distribution) {
  chart.data.labels = Object.keys(distribution);
  chart.data.datasets[0].data = Object.values(distribution);
  chart.update('active');
}

// ── Attack Trend Area Chart ──────────────────────────────────────────────────

export function buildTrendChart(canvasId, trendsData) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const { hours, series } = trendsData;

  const datasets = Object.entries(series).slice(0, 5).map(([label, data], i) => {
    const color = PALETTE[i % PALETTE.length];
    return {
      label,
      data,
      borderColor: color,
      backgroundColor: color + '18',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: true,
      tension: 0.4,
    };
  });

  return new Chart(ctx, {
    type: 'line',
    data: { labels: hours, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          grid: { color: 'rgba(0,212,255,0.04)' },
          ticks: { maxTicksLimit: 8 }
        },
        y: {
          grid: { color: 'rgba(0,212,255,0.04)' },
          ticks: { callback: v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v }
        }
      },
      plugins: { legend: { position: 'top' } },
      animation: { duration: 800 }
    }
  });
}

export function updateTrendChart(chart, trendsData) {
  const { hours, series } = trendsData;
  chart.data.labels = hours;
  chart.data.datasets = Object.entries(series).slice(0, 5).map(([label, data], i) => {
    const color = PALETTE[i % PALETTE.length];
    return {
      label, data,
      borderColor: color, backgroundColor: color + '18',
      borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
      fill: true, tension: 0.4,
    };
  });
  chart.update();
}

// ── Feature Importance Horizontal Bar ───────────────────────────────────────

export function buildFeaturesChart(canvasId, featuresData) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const sorted = [...featuresData].sort((a, b) => b.importance - a.importance).slice(0, 15);
  const labels = sorted.map(f => f.feature);
  const data   = sorted.map(f => +(f.importance * 100).toFixed(3));

  const gradient = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
  gradient.addColorStop(0, '#00d4ff');
  gradient.addColorStop(1, '#a855f7');

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance (%)',
        data,
        backgroundColor: gradient,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(0,212,255,0.04)' },
          ticks: { callback: v => `${v.toFixed(1)}%` }
        },
        y: { grid: { display: false }, ticks: { font: { size: 10 } } }
      },
      plugins: { legend: { display: false } },
      animation: { duration: 700 }
    }
  });
}

export function updateFeaturesChart(chart, featuresData) {
  const sorted = [...featuresData].sort((a, b) => b.importance - a.importance).slice(0, 15);
  chart.data.labels = sorted.map(f => f.feature);
  chart.data.datasets[0].data = sorted.map(f => +(f.importance * 100).toFixed(3));
  chart.update();
}

// ── Cluster Radar / Bubble ───────────────────────────────────────────────────

export function buildClusterChart(canvasId, clusters) {
  const ctx = document.getElementById(canvasId).getContext('2d');

  const data = clusters.map((c, i) => ({
    x: i,
    y: c.size,
    r: Math.max(6, Math.sqrt(c.size / 200)),
    label: c.dominant_attack,
  }));

  return new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [{
        label: 'Clusters',
        data,
        backgroundColor: clusters.map((_, i) => PALETTE[i % PALETTE.length] + '88'),
        borderColor:     clusters.map((_, i) => PALETTE[i % PALETTE.length]),
        borderWidth: 1.5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Cluster ID' },
          grid: { color: 'rgba(0,212,255,0.04)' }
        },
        y: {
          title: { display: true, text: 'Flow Count' },
          grid: { color: 'rgba(0,212,255,0.04)' },
          ticks: { callback: v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return [`Cluster ${d.x}`, `Attack: ${d.label}`, `Flows: ${d.y.toLocaleString()}`];
            }
          }
        }
      },
      animation: { duration: 700 }
    }
  });
}

// ── Top Targeted Ports Bar Chart ─────────────────────────────────────────────

export function buildPortChart(canvasId, portData) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const labels = Object.keys(portData);
  const data = Object.values(portData);

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Flow Count',
        data,
        backgroundColor: labels.map((_, i) => PALETTE[i % PALETTE.length] + 'b0'),
        borderColor: labels.map((_, i) => PALETTE[i % PALETTE.length]),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(0,212,255,0.04)' },
          title: { display: true, text: 'Target Port' }
        },
        y: {
          grid: { color: 'rgba(0,212,255,0.04)' },
          ticks: { callback: v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// ── Protocol & Flag Radar Chart ──────────────────────────────────────────────

export function buildProtocolRadarChart(canvasId, radarData) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'radar',
    data: {
      labels: radarData.labels || ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG', 'ECE'],
      datasets: [
        {
          label: 'Attack Traffic',
          data: radarData.attack || [85, 92, 40, 60, 75, 20, 15],
          borderColor: '#ff4757',
          backgroundColor: 'rgba(255, 71, 87, 0.2)',
          borderWidth: 2,
          pointBackgroundColor: '#ff4757',
        },
        {
          label: 'Benign Traffic',
          data: radarData.benign || [30, 80, 50, 10, 45, 5, 2],
          borderColor: '#2ed573',
          backgroundColor: 'rgba(46, 213, 115, 0.2)',
          borderWidth: 2,
          pointBackgroundColor: '#2ed573',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(0,212,255,0.1)' },
          grid: { color: 'rgba(0,212,255,0.08)' },
          pointLabels: { color: '#a8c4e0', font: { size: 10 } },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10 } }
      }
    }
  });
}

