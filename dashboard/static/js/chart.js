/**
 * static/js/chart.js
 */

const CHART = (() => {
  const N = 30;

  let awsSeries   = _genSeries(0.195, 0.018, N);
  let gcpSeries   = _genSeries(0.210, 0.010, N);
  let azureSeries = _genSeries(0.200, 0.014, N);
  let labels      = _defaultLabels(N);
  let _lastParetoData = null;  // ← add this

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

  function _cssVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }

 function applyHistory(data) {
  if (!data || !data.timestamps || data.timestamps.length === 0) return false;
 
  function interp(arr) {
    if (!arr || arr.length === 0) return null;
    const out = arr.slice();
 
    // Forward-fill: carry the last known real value forward
    for (let i = 1; i < out.length; i++) {
      if (out[i] === null && out[i - 1] !== null) out[i] = out[i - 1];
    }
    // Backward-fill: carry the first known real value backward (leading nulls)
    for (let i = out.length - 2; i >= 0; i--) {
      if (out[i] === null && out[i + 1] !== null) out[i] = out[i + 1];
    }
 
    // If still entirely null (cloud had no data at all), signal caller
    return out.every(v => v === null) ? null : out;
  }
 
  const aws   = interp(data.aws);
  const gcp   = interp(data.gcp);
  const azure = interp(data.azure);
 
  // Only swap out the series if we actually got real data for that cloud.
  // Remaining nulls after interp can't happen (interp fills all), but guard anyway.
  if (aws)   awsSeries   = aws.map(v => v ?? awsSeries[awsSeries.length - 1]);
  if (gcp)   gcpSeries   = gcp.map(v => v ?? gcpSeries[gcpSeries.length - 1]);
  if (azure) azureSeries = azure.map(v => v ?? azureSeries[azureSeries.length - 1]);
 
  labels = data.timestamps.slice(-N);
  return true;
}
 

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

  function getCurrentValues() {
    const last = arr => arr[arr.length - 1];
    const prev = arr => arr[arr.length - 2] ?? arr[arr.length - 1];
    const min  = arr => Math.min(...arr);
    const max  = arr => Math.max(...arr);
    const avg  = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    return {
      aws:   { cur: last(awsSeries),   prev: prev(awsSeries),   min: min(awsSeries),   max: max(awsSeries),   avg: avg(awsSeries),   series: [...awsSeries]   },
      gcp:   { cur: last(gcpSeries),   prev: prev(gcpSeries),   min: min(gcpSeries),   max: max(gcpSeries),   avg: avg(gcpSeries),   series: [...gcpSeries]   },
      azure: { cur: last(azureSeries), prev: prev(azureSeries), min: min(azureSeries), max: max(azureSeries), avg: avg(azureSeries), series: [...azureSeries] },
    };
  }

  function drawPriceChart() {
    const canvas = document.getElementById('price-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

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

    const mutedColor = _cssVar('--muted') || '#718096';
    const isLight    = document.documentElement.classList.contains('light');
    const gridColor  = isLight
      ? 'rgba(43,108,176,0.08)'
      : 'rgba(99,179,237,0.07)';

    ctx.strokeStyle = gridColor;
    ctx.lineWidth   = 1;
    const gridLines = 6;
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

    ctx.fillStyle = mutedColor;
    ctx.font      = '9px IBM Plex Mono,monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i += 5) {
      ctx.fillText(labels[i] || '', tx(i), H - 8);
    }

    const series = [
      { data: awsSeries,   color: _cssVar('--aws')   || (isLight ? '#c05621' : '#f6ad55') },
      { data: gcpSeries,   color: _cssVar('--gcp')   || (isLight ? '#276749' : '#68d391') },
      { data: azureSeries, color: _cssVar('--azure') || (isLight ? '#2b6cb0' : '#63b3ed') },
    ];

    series.forEach(s => {
      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => ctx.lineTo(tx(i), ty(v)));
      ctx.lineTo(tx(n - 1), pad.top + ch);
      ctx.lineTo(tx(0),     pad.top + ch);
      ctx.closePath();
      ctx.fillStyle = s.color + (isLight ? '20' : '18');
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(tx(0), ty(s.data[0]));
      s.data.forEach((v, i) => { if (i > 0) ctx.lineTo(tx(i), ty(v)); });
      ctx.strokeStyle = s.color;
      ctx.lineWidth   = 2;
      ctx.lineJoin    = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(tx(n - 1), ty(s.data[n - 1]), 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
    });

    drawSparklines();
  }

  /**
   * Draw mini sparklines inside the cloud health strip.
   * Reads from the same series arrays so no extra data needed.
   * Uses last 30 points (same N as main chart).
   */
  function drawSparklines() {
    const isLight = document.documentElement.classList.contains('light');
    const clouds = [
      { id: 'sp-aws',   data: awsSeries,   color: _cssVar('--aws')   || (isLight ? '#c05621' : '#f6ad55') },
      { id: 'sp-gcp',   data: gcpSeries,   color: _cssVar('--gcp')   || (isLight ? '#276749' : '#68d391') },
      { id: 'sp-azure', data: azureSeries, color: _cssVar('--azure') || (isLight ? '#2b6cb0' : '#63b3ed') },
    ];

    clouds.forEach(({ id, data, color }) => {
      const c = document.getElementById(id);
      if (!c) return;
      const wrap = c.parentElement;
      const W = wrap ? wrap.offsetWidth : 160;
      const H = 28;
      c.width  = W;
      c.height = H;

      const n  = data.length;
      if (n < 2) return;
      const mn = Math.min(...data) * 0.98;
      const mx = Math.max(...data) * 1.02;
      const rng = mx - mn || 0.001;

      const tx = i => (i / (n - 1)) * (W - 2) + 1;
      const ty = v => H - 2 - ((v - mn) / rng) * (H - 4);

      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, W, H);

      ctx.beginPath();
      data.forEach((v, i) => i === 0 ? ctx.moveTo(tx(i), ty(v)) : ctx.lineTo(tx(i), ty(v)));
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.lineJoin    = 'round';
      ctx.stroke();

      ctx.lineTo(tx(n - 1), H);
      ctx.lineTo(tx(0), H);
      ctx.closePath();
      ctx.fillStyle = color + '22';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(tx(n - 1), ty(data[n - 1]), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

function drawPareto(apiData) {
  if (apiData && apiData.pareto_set) _lastParetoData = apiData;
  const data = _lastParetoData;

  const canvas = document.getElementById('pareto-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  const pad     = { top: 14, right: 16, bottom: 36, left: 48 };
  const cw      = W - pad.left - pad.right;
  const ch      = H - pad.top  - pad.bottom;
  const isLight = document.documentElement.classList.contains('light');
  const muted   = _cssVar('--muted') || '#718096';
  const gridCol = isLight ? 'rgba(43,108,176,0.08)' : 'rgba(99,179,237,0.07)';
  const colors  = {
    aws:   _cssVar('--aws')   || (isLight ? '#c05621' : '#f6ad55'),
    gcp:   _cssVar('--gcp')   || (isLight ? '#276749' : '#68d391'),
    azure: _cssVar('--azure') || (isLight ? '#2b6cb0' : '#63b3ed'),
  };

  // ── Points ──────────────────────────────────────────────────────
  let points = [];
  let winner = null;

  if (data && data.pareto_set && data.pareto_set.length > 0) {
    points = data.pareto_set.map(p => ({
      cloud:  p.cloud,
      cost:   p.est_cost,
      risk:   p.preemption_risk,
      label:  p.instance_type.replace('Standard_', '').replace('-standard-', '-'),
      pareto: true,
    }));
    winner = data.winner;
    const badge = document.getElementById('pareto-badge');
    if (badge) {
      badge.textContent = `Pareto · ${points.length} pts · ${new Date().toTimeString().slice(0,5)}`;
      badge.className   = 'badge badge-live';
    }
  } else {
    points = [
      { cost: 0.30, risk: 0.45, cloud: 'aws',   pareto: true,  label: 'g4dn.xlarge'   },
      { cost: 0.18, risk: 0.30, cloud: 'aws',   pareto: false, label: 'g4dn.medium'   },
      { cost: 0.42, risk: 0.20, cloud: 'gcp',   pareto: true,  label: 'g2-std-4'      },
      { cost: 0.55, risk: 0.15, cloud: 'gcp',   pareto: false, label: 'a2-highgpu'    },
      { cost: 0.25, risk: 0.50, cloud: 'azure', pareto: false, label: 'NC4as_T4'      },
      { cost: 0.48, risk: 0.25, cloud: 'azure', pareto: true,  label: 'NC6s_v3'       },
    ];
  }

  // ── Axis ranges ─────────────────────────────────────────────────
  const allCosts = points.map(p => p.cost);
  const allRisks = points.map(p => p.risk);
  const maxCost  = Math.max(...allCosts) * 1.20 || 1;
  const maxRisk  = Math.max(...allRisks) * 1.20 || 1;
  const midCost  = maxCost / 2;
  const midRisk  = maxRisk / 2;

  const tx = c => pad.left + (c / maxCost) * cw;
  const ty = r => pad.top  + ch - (r / maxRisk) * ch;

  // ── Quadrant shading ────────────────────────────────────────────
  const qAlpha = isLight ? '0a' : '0f';
  const mx = tx(midCost);
  const my = ty(midRisk);

  // Bottom-left = OPTIMAL (green tint)
  ctx.fillStyle = (isLight ? '#276749' : '#68d391') + qAlpha;
  ctx.fillRect(pad.left, my, mx - pad.left, pad.top + ch - my);

  // Top-right = AVOID (red tint)
  ctx.fillStyle = (isLight ? '#9b2c2c' : '#fc8181') + qAlpha;
  ctx.fillRect(mx, pad.top, pad.left + cw - mx, my - pad.top);

  // Quadrant dividers
  ctx.strokeStyle = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(mx, pad.top); ctx.lineTo(mx, pad.top + ch); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.left, my); ctx.lineTo(pad.left + cw, my); ctx.stroke();
  ctx.setLineDash([]);

  // Quadrant labels
  ctx.font      = '7px IBM Plex Mono,monospace';
  ctx.fillStyle = (isLight ? '#276749' : '#68d391') + '99';
  ctx.textAlign = 'left';
  ctx.fillText('OPTIMAL', pad.left + 4, pad.top + ch - 4);
  ctx.fillStyle = (isLight ? '#9b2c2c' : '#fc8181') + '99';
  ctx.textAlign = 'right';
  ctx.fillText('AVOID', pad.left + cw - 4, pad.top + 10);

  // ── Grid lines + axis labels ────────────────────────────────────
  ctx.strokeStyle = gridCol;
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y   = pad.top + (i / 4) * ch;
    const val = maxRisk - (i / 4) * maxRisk;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    ctx.fillStyle = muted;
    ctx.font      = '8px IBM Plex Mono,monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(2), pad.left - 5, y + 3);

    const x    = pad.left + (i / 4) * cw;
    const cval = (i / 4) * maxCost;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ch); ctx.stroke();
    ctx.fillStyle = muted;
    ctx.textAlign = 'center';
    ctx.fillText('$' + cval.toFixed(2), x, pad.top + ch + 12);
  }

  ctx.fillStyle = muted;
  ctx.font      = '8px IBM Plex Mono,monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Est. Cost ($)', pad.left + cw / 2, H - 4);
  ctx.save();
  ctx.translate(10, pad.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Risk', 0, 0);
  ctx.restore();

  // ── Pareto frontier step-line ───────────────────────────────────
  // A true Pareto frontier is a staircase (dominance is L-shaped)
  const pf = points.filter(p => p.pareto).sort((a, b) => a.cost - b.cost);
  if (pf.length > 1) {
    ctx.strokeStyle = isLight ? 'rgba(0,0,0,0.18)' : 'rgba(255,255,255,0.18)';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(tx(pf[0].cost), ty(pf[0].risk));
    for (let i = 1; i < pf.length; i++) {
      // Step: go right first, then up — classic Pareto staircase
      ctx.lineTo(tx(pf[i].cost), ty(pf[i - 1].risk));
      ctx.lineTo(tx(pf[i].cost), ty(pf[i].risk));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── Points ──────────────────────────────────────────────────────
  // Spread overlapping points with jitter before drawing
  const JITTER  = 12; // px — max nudge to separate overlapping labels
  const placed  = [];

  function _nudge(x, y) {
    for (const p of placed) {
      if (Math.abs(p.x - x) < JITTER && Math.abs(p.y - y) < JITTER) {
        return { x: x + (Math.random() - 0.5) * JITTER,
                 y: y + (Math.random() - 0.5) * JITTER };
      }
    }
    return { x, y };
  }

  points.forEach(p => {
    let { x, y } = _nudge(tx(p.cost), ty(p.risk));
    placed.push({ x, y });

    const color = colors[p.cloud] || '#aaa';
    const isWinner = winner &&
      p.cloud === winner.cloud &&
      p.label.startsWith(winner.instance_type.replace('Standard_','').split('-standard-')[0].slice(0,6));

    // Outer ring for pareto points
    if (p.pareto) {
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = color + 'aa';
      ctx.lineWidth   = 1;
      ctx.stroke();
    }

    // Dot
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.pareto ? color : color + '55';
    ctx.fill();

    // Winner glow + star
    if (isWinner) {
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.stroke();
      ctx.font      = '10px IBM Plex Mono,monospace';
      ctx.fillStyle = color;
      ctx.textAlign = 'center';
      ctx.fillText('★', x, y - 14);
    }

    // Instance label — alternate above/below to reduce overlap
    const labelY = (placed.length % 2 === 0) ? y + 14 : y - 8;
    ctx.font      = '7px IBM Plex Mono,monospace';
    ctx.fillStyle = muted;
    ctx.textAlign = 'center';
    ctx.fillText(p.label, x, labelY);
  });
}

  return { applyHistory, mockTick, getCurrentValues, drawPriceChart, drawSparklines, drawPareto };
})();