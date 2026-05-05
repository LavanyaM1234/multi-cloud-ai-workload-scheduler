/**
 * static/js/chart.js
 * ──────────────────
 * Handles all canvas rendering:
 *   - Price history line chart (3 clouds)
 *   - Pareto frontier scatter plot (cost vs time)
 *
 * Called by api.js whenever fresh data arrives.
 * Also has a mock tick function for when no API is available.
 */

const CHART = (() => {
  const N = 30; // number of data points shown on price chart

  // ── Internal state ───────────────────────────────────────────────
  let awsSeries   = _genSeries(0.195, 0.018, N);
  let gcpSeries   = _genSeries(0.210, 0.010, N);
  let azureSeries = _genSeries(0.200, 0.014, N);
  let labels      = _defaultLabels(N);

  // ── Mock data generators ─────────────────────────────────────────
  function _genSeries(base, vol, n) {
    const arr = [base];
    for (let i = 1; i < n; i++) {
      const spike = Math.random() < 0.10 ? Math.random() * 0.08 : 0;
      const drop  = Math.random() < 0.10 ? -(Math.random() * 0.06) : 0;
      arr.push(Math.max(0.05, +(arr[i-1] + (Math.random()-0.48)*vol + spike + drop).toFixed(4)));
    }
    return arr;
  }

  function _defaultLabels(n) {
    return Array.from({ length: n }, (_, i) => {
      const d = new Date();
      d.setMinutes(d.getMinutes() - (n - 1 - i));
      return d.toTimeString().slice(0, 5);
    });
  }

  // ── Apply real data from API ─────────────────────────────────────
  function applyHistory(data) {
    if (!data || !data.timestamps || data.timestamps.length === 0) return false;
    const fill = (arr, fallback) =>
      arr.map((v, i) => v !== null ? v : fallback[i] ?? fallback[fallback.length-1]);
    awsSeries   = fill(data.aws,   awsSeries);
    gcpSeries   = fill(data.gcp,   gcpSeries);
    azureSeries = fill(data.azure, azureSeries);
    labels      = data.timestamps.slice(-N);
    return true;
  }

  // ── Mock tick — keeps chart moving when no API ───────────────────
  function mockTick() {
    const addPt = (arr, vol) => {
      const last  = arr[arr.length - 1];
      const spike = Math.random() < 0.08 ? Math.random() * 0.05 : 0;
      const drop  = Math.random() < 0.08 ? -(Math.random() * 0.04) : 0;
      arr.push(Math.max(0.05, +(last + (Math.random()-0.47)*vol + spike + drop).toFixed(4)));
      if (arr.length > N) arr.shift();
    };
    addPt(awsSeries,   0.018);
    addPt(gcpSeries,   0.010);
    addPt(azureSeries, 0.014);
  }

  // ── Get current values (for numeric cards + stat cards) ──────────
  function getCurrentValues() {
    const last  = arr => arr[arr.length - 1];
    const prev  = arr => arr[arr.length - 2] ?? arr[arr.length - 1];
    const min   = arr => Math.min(...arr);
    const max   = arr => Math.max(...arr);
    const avg   = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    return {
      aws:   { cur: last(awsSeries),   prev: prev(awsSeries),   min: min(awsSeries),   max: max(awsSeries),   avg: avg(awsSeries) },
      gcp:   { cur: last(gcpSeries),   prev: prev(gcpSeries),   min: min(gcpSeries),   max: max(gcpSeries),   avg: avg(gcpSeries) },
      azure: { cur: last(azureSeries), prev: prev(azureSeries), min: min(azureSeries), max: max(azureSeries), avg: avg(azureSeries) },
    };
  }

  // ── Draw price chart ─────────────────────────────────────────────
  function drawPriceChart() {
    const canvas = document.getElementById('price-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;

    const pad = { top: 12, right: 16, bottom: 28, left: 52 };
    const cw  = W - pad.left - pad.right;
    const ch  = H - pad.top  - pad.bottom;

    const allVals = [...awsSeries, ...gcpSeries, ...azureSeries];
    const minV    = Math.min(...allVals) * 0.97;
    const maxV    = Math.max(...allVals) * 1.03;
    const n       = awsSeries.length;

    const tx = i => pad.left + (i / (n-1)) * cw;
    const ty = v => pad.top  + ch - ((v - minV) / (maxV - minV)) * ch;

    // Grid lines + Y labels
    ctx.strokeStyle = 'rgba(99,179,237,0.07)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
      const y   = pad.top + (i/4) * ch;
      const val = maxV - (i/4) * (maxV - minV);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
      ctx.fillStyle   = '#4a5568';
      ctx.font        = '9px IBM Plex Mono,monospace';
      ctx.textAlign   = 'right';
      ctx.fillText('$' + val.toFixed(3), pad.left - 6, y + 3);
    }

    // X labels
    ctx.fillStyle = '#4a5568';
    ctx.font      = '9px IBM Plex Mono,monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i += 5) {
      ctx.fillText(labels[i] || '', tx(i), H - 6);
    }

    // Draw each cloud series
    const series = [
      { data: awsSeries,   color: '#f6ad55' },
      { data: gcpSeries,   color: '#68d391' },
      { data: azureSeries, color: '#63b3ed' },
    ];

    series.forEach(s => {
      // Filled area
      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => ctx.lineTo(tx(i), ty(v)));
      ctx.lineTo(tx(n-1), pad.top + ch);
      ctx.lineTo(tx(0),   pad.top + ch);
      ctx.closePath();
      ctx.fillStyle = s.color + '18';
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => { if (i > 0) ctx.lineTo(tx(i), ty(v)); });
      ctx.strokeStyle = s.color;
      ctx.lineWidth   = 1.8;
      ctx.lineJoin    = 'round';
      ctx.stroke();

      // Endpoint dot
      ctx.beginPath();
      ctx.arc(tx(n-1), ty(s.data[n-1]), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
    });
  }

  // ── Draw Pareto chart ────────────────────────────────────────────
  function drawPareto(paretoPoints) {
    const canvas = document.getElementById('pareto-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;

    const pad    = { top: 10, right: 10, bottom: 28, left: 40 };
    const cw     = W - pad.left - pad.right;
    const ch     = H - pad.top  - pad.bottom;
    const maxC   = 0.7;
    const maxT   = 9;
    const colors = { aws: '#f6ad55', gcp: '#68d391', azure: '#63b3ed' };

    const tx = t => pad.left + (t / maxT) * cw;
    const ty = c => pad.top  + ch - (c / maxC) * ch;

    // Use provided points or fallback to mock
    const points = paretoPoints || [
      { cost: 0.30, time: 4.0, cloud: 'aws',   pareto: true  },
      { cost: 0.18, time: 6.5, cloud: 'aws',   pareto: true  },
      { cost: 0.42, time: 3.2, cloud: 'gcp',   pareto: true  },
      { cost: 0.55, time: 2.8, cloud: 'gcp',   pareto: false },
      { cost: 0.25, time: 5.1, cloud: 'azure', pareto: false },
      { cost: 0.48, time: 2.5, cloud: 'azure', pareto: true  },
    ];

    // Grid
    ctx.strokeStyle = 'rgba(99,179,237,0.07)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 3; i++) {
      const y = pad.top + (i/3) * ch;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
      ctx.fillStyle   = '#4a5568';
      ctx.font        = '8px IBM Plex Mono,monospace';
      ctx.textAlign   = 'right';
      ctx.fillText('$' + (maxC - (i/3)*maxC).toFixed(2), pad.left - 4, y + 3);
    }
    ctx.fillStyle = '#4a5568'; ctx.textAlign = 'center';
    ctx.fillText('Time (hrs)', pad.left + cw/2, H - 2);

    // Pareto frontier dashed line
    const pf = points.filter(p => p.pareto).sort((a, b) => a.time - b.time);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    pf.forEach((p, i) => { i === 0 ? ctx.moveTo(tx(p.time), ty(p.cost)) : ctx.lineTo(tx(p.time), ty(p.cost)); });
    ctx.stroke();
    ctx.setLineDash([]);

    // Points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(tx(p.time), ty(p.cost), p.pareto ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.pareto ? colors[p.cloud] : colors[p.cloud] + '44';
      ctx.fill();
      if (p.pareto) {
        ctx.strokeStyle = colors[p.cloud];
        ctx.lineWidth   = 1.5;
        ctx.stroke();
      }
    });
  }

  // ── Public API ───────────────────────────────────────────────────
  return { applyHistory, mockTick, getCurrentValues, drawPriceChart, drawPareto };
})();
