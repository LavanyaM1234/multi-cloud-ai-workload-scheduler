/**
 * static/js/chart.js
 * ──────────────────
 * Handles all canvas rendering:
 *   - Price history line chart (3 clouds) — full panel width
 *   - Pareto frontier scatter plot (cost vs time)
 *
 * Changes from original:
 *   - drawPriceChart() reads theme-aware colours from CSS variables
 *     so grid lines + labels look correct in both dark and light mode
 *   - drawPareto() same fix for frontier line colour
 *   - Chart fills full panel width at any height set in CSS
 */

const CHART = (() => {
  const N = 30;

  // ── Internal state ────────────────────────────────────────────────
  let awsSeries   = _genSeries(0.195, 0.018, N);
  let gcpSeries   = _genSeries(0.210, 0.010, N);
  let azureSeries = _genSeries(0.200, 0.014, N);
  let labels      = _defaultLabels(N);

  // ── Helpers ───────────────────────────────────────────────────────

  function _genSeries(base, vol, n) {
    const arr = [base];
    for (let i = 1; i < n; i++) {
      const spike = Math.random() < 0.10 ? Math.random() * 0.08 : 0;
      const drop  = Math.random() < 0.10 ? -(Math.random() * 0.06) : 0;
      arr.push(Math.max(0.05,
        +(arr[i-1] + (Math.random()-0.48)*vol + spike + drop).toFixed(4)));
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

  /**
   * Read a CSS variable from the document root.
   * Used so chart colours respond to dark/light theme toggle.
   */
  function _cssVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }

  // ── Apply real API data ───────────────────────────────────────────

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

  // ── Mock tick — keeps chart moving without API ────────────────────

  function mockTick() {
    const addPt = (arr, vol) => {
      const last  = arr[arr.length - 1];
      const spike = Math.random() < 0.08 ? Math.random() * 0.05 : 0;
      const drop  = Math.random() < 0.08 ? -(Math.random() * 0.04) : 0;
      arr.push(Math.max(0.05,
        +(last + (Math.random()-0.47)*vol + spike + drop).toFixed(4)));
      if (arr.length > N) arr.shift();
    };
    addPt(awsSeries,   0.018);
    addPt(gcpSeries,   0.010);
    addPt(azureSeries, 0.014);
  }

  // ── Current values for stat + numeric cards ───────────────────────

  function getCurrentValues() {
    const last = arr => arr[arr.length - 1];
    const prev = arr => arr[arr.length - 2] ?? arr[arr.length - 1];
    const min  = arr => Math.min(...arr);
    const max  = arr => Math.max(...arr);
    const avg  = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    return {
      aws:   { cur: last(awsSeries),   prev: prev(awsSeries),   min: min(awsSeries),   max: max(awsSeries),   avg: avg(awsSeries)   },
      gcp:   { cur: last(gcpSeries),   prev: prev(gcpSeries),   min: min(gcpSeries),   max: max(gcpSeries),   avg: avg(gcpSeries)   },
      azure: { cur: last(azureSeries), prev: prev(azureSeries), min: min(azureSeries), max: max(azureSeries), avg: avg(azureSeries) },
    };
  }

  // ── Draw price chart — fills full .chart-wrap width/height ────────

  function drawPriceChart() {
    const canvas = document.getElementById('price-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Use the wrapper's actual rendered size so the canvas fills it fully
    const wrap = canvas.parentElement;
    const W = wrap ? wrap.offsetWidth  : canvas.offsetWidth;
    const H = wrap ? wrap.offsetHeight : canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;

    const pad = { top: 16, right: 20, bottom: 32, left: 58 };
    const cw  = W - pad.left - pad.right;
    const ch  = H - pad.top  - pad.bottom;

    const allVals = [...awsSeries, ...gcpSeries, ...azureSeries];
    const minV    = Math.min(...allVals) * 0.97;
    const maxV    = Math.max(...allVals) * 1.03;
    const n       = awsSeries.length;

    const tx = i => pad.left + (i / (n - 1)) * cw;
    const ty = v => pad.top  + ch - ((v - minV) / (maxV - minV)) * ch;

    // Theme-aware colours — read from CSS variables
    const mutedColor  = _cssVar('--muted')  || '#718096';
    const gridColor   = document.documentElement.classList.contains('light')
      ? 'rgba(43,108,176,0.08)'
      : 'rgba(99,179,237,0.07)';

    // ── Grid lines + Y labels ───────────────────────────────────────
    ctx.strokeStyle = gridColor;
    ctx.lineWidth   = 1;
    const gridLines = 6;   // more lines = better resolution at taller height
    for (let i = 0; i <= gridLines; i++) {
      const y   = pad.top + (i / gridLines) * ch;
      const val = maxV - (i / gridLines) * (maxV - minV);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
      ctx.fillStyle = mutedColor;
      ctx.font      = '9px IBM Plex Mono,monospace';
      ctx.textAlign = 'right';
      ctx.fillText('$' + val.toFixed(3), pad.left - 6, y + 3);
    }

    // ── X labels (every 5 points) ───────────────────────────────────
    ctx.fillStyle = mutedColor;
    ctx.font      = '9px IBM Plex Mono,monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i += 5) {
      ctx.fillText(labels[i] || '', tx(i), H - 8);
    }

    // ── Cloud series ────────────────────────────────────────────────
    // Colours come from CSS variables so they work in light theme too
    const isLight = document.documentElement.classList.contains('light');
    const series = [
      { data: awsSeries,   color: _cssVar('--aws')   || (isLight ? '#c05621' : '#f6ad55') },
      { data: gcpSeries,   color: _cssVar('--gcp')   || (isLight ? '#276749' : '#68d391') },
      { data: azureSeries, color: _cssVar('--azure') || (isLight ? '#2b6cb0' : '#63b3ed') },
    ];

    series.forEach(s => {
      // Filled area under line
      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => ctx.lineTo(tx(i), ty(v)));
      ctx.lineTo(tx(n - 1), pad.top + ch);
      ctx.lineTo(tx(0),     pad.top + ch);
      ctx.closePath();
      ctx.fillStyle = s.color + (isLight ? '20' : '18');
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => { if (i > 0) ctx.lineTo(tx(i), ty(v)); });
      ctx.strokeStyle = s.color;
      ctx.lineWidth   = 2;
      ctx.lineJoin    = 'round';
      ctx.stroke();

      // Endpoint dot
      ctx.beginPath();
      ctx.arc(tx(n - 1), ty(s.data[n - 1]), 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
    });
  }

  // ── Draw Pareto chart ─────────────────────────────────────────────

  function drawPareto(paretoPoints) {
    const canvas = document.getElementById('pareto-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;

    const pad  = { top: 10, right: 10, bottom: 28, left: 40 };
    const cw   = W - pad.left - pad.right;
    const ch   = H - pad.top  - pad.bottom;
    const maxC = 0.7;
    const maxT = 9;

    const isLight   = document.documentElement.classList.contains('light');
    const mutedColor = _cssVar('--muted') || '#718096';
    const gridColor  = isLight ? 'rgba(43,108,176,0.08)' : 'rgba(99,179,237,0.07)';
    const frontierColor = isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.12)';

    const colors = {
      aws:   _cssVar('--aws')   || (isLight ? '#c05621' : '#f6ad55'),
      gcp:   _cssVar('--gcp')   || (isLight ? '#276749' : '#68d391'),
      azure: _cssVar('--azure') || (isLight ? '#2b6cb0' : '#63b3ed'),
    };

    const tx = t => pad.left + (t / maxT) * cw;
    const ty = c => pad.top  + ch - (c / maxC) * ch;

    const points = paretoPoints || [
      { cost: 0.30, time: 4.0, cloud: 'aws',   pareto: true  },
      { cost: 0.18, time: 6.5, cloud: 'aws',   pareto: true  },
      { cost: 0.42, time: 3.2, cloud: 'gcp',   pareto: true  },
      { cost: 0.55, time: 2.8, cloud: 'gcp',   pareto: false },
      { cost: 0.25, time: 5.1, cloud: 'azure', pareto: false },
      { cost: 0.48, time: 2.5, cloud: 'azure', pareto: true  },
    ];

    // Grid
    ctx.strokeStyle = gridColor;
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 3; i++) {
      const y = pad.top + (i / 3) * ch;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
      ctx.fillStyle = mutedColor;
      ctx.font      = '8px IBM Plex Mono,monospace';
      ctx.textAlign = 'right';
      ctx.fillText('$' + (maxC - (i / 3) * maxC).toFixed(2), pad.left - 4, y + 3);
    }
    ctx.fillStyle = mutedColor;
    ctx.textAlign = 'center';
    ctx.fillText('Time (hrs)', pad.left + cw / 2, H - 2);

    // Pareto frontier dashed line — theme-aware colour
    const pf = points.filter(p => p.pareto).sort((a, b) => a.time - b.time);
    ctx.strokeStyle = frontierColor;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    pf.forEach((p, i) => {
      i === 0 ? ctx.moveTo(tx(p.time), ty(p.cost))
              : ctx.lineTo(tx(p.time), ty(p.cost));
    });
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

  return { applyHistory, mockTick, getCurrentValues, drawPriceChart, drawPareto };
})();