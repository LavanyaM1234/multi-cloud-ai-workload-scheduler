/**
 * static/js/ui.js
 *
 * Changes vs previous version:
 *   [1] updateNumericCards() and updateNumericCardsMock() removed —
 *       replaced by updateHealthStrip() which drives the new cloud
 *       health strip below the price chart.
 *   [2] updateStats() now writes to both the stat card (stat-ckpt-total,
 *       stat-ckpt-emergency) and the jobs panel (ckpt-total, ckpt-emergency)
 *       to fix the duplicate-ID bug that prevented the stat card from updating.
 *   [3] updateHealthStrip() computes delta %, volatility label, and
 *       best-value badge from CHART.getCurrentValues() shape.
 */

const UI = (() => {

  // ── Clock ─────────────────────────────────────────────────────────
  function startClock() {
    const tick = () => {
      const el = document.getElementById('clock');
      if (el) el.textContent = new Date().toUTCString().slice(17, 25) + ' UTC';
    };
    tick();
    setInterval(tick, 1000);
  }

  // ── Stat cards (top row) ──────────────────────────────────────────
  function updateStatCards(summary) {
    if (!summary) return;
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const d = summary[cloud]; if (!d) return;
      const cur = parseFloat(d.current_price);
      const avg = parseFloat(d.avg_price);
      const pct = ((cur - avg) / avg * 100).toFixed(1);
      const pEl = document.getElementById(cloud + '-price');
      const dEl = document.getElementById(cloud + '-delta');
      if (pEl) pEl.textContent = '$' + cur.toFixed(3);
      if (dEl) {
        dEl.textContent = (pct >= 0 ? '▲ ' : '▼ ') + Math.abs(pct) + '% vs 3hr avg';
        dEl.className   = 'stat-delta ' + (pct >= 0 ? 'delta-up' : 'delta-down');
      }
    });
  }

  // ── Cloud health strip (below price chart) ────────────────────────
  /**
   * Updates the three health cards with current price, delta pill,
   * sparklines (drawn by CHART.drawSparklines), volatility label,
   * min/max/avg, and the best-value badge.
   *
   * values = CHART.getCurrentValues() — shape:
   *   { aws, gcp, azure } each with { cur, prev, min, max, avg, series }
   */
  function updateHealthStrip(values, apiMeta, summary) {
  if (!values) return;
 
  const clouds = ['aws', 'gcp', 'azure'];
  const bestCloud = clouds.reduce((a, b) =>
    (values[a]?.cur ?? Infinity) < (values[b]?.cur ?? Infinity) ? a : b
  );
 
  clouds.forEach(cloud => {
    const v = values[cloud];
    if (!v) return;
 
    const meta   = apiMeta?.[cloud];
    const sumRow = summary?.[cloud];          // from /api/prices/summary
 
    // Prefer summary current_price (clean BQ query) over series last value
    const currentPrice = sumRow?.current_price ?? v.cur;
    const prevPrice    = v.prev;
 
    const fmt  = n  => '$' + n.toFixed(4);
    const pct  = ((currentPrice - prevPrice) / (prevPrice || 1) * 100).toFixed(1);
    const isUp = parseFloat(pct) >= 0;
 
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
 
    // Instance label from API meta
    if (meta?.instance) {
      const cloudNames = { aws: 'AWS', gcp: 'GCP', azure: 'Azure' };
      const labelEl = document.querySelector(`.${cloud}-hcard .hcard-cloud`);
      if (labelEl) labelEl.textContent = `${cloudNames[cloud]} · ${meta.instance}`;
    }
 
    // Current price — 4dp so small moves are visible
    const curEl = document.getElementById('h' + cloud + '-cur');
    if (curEl) curEl.textContent = '$' + currentPrice.toFixed(4);
 
    const dEl = document.getElementById('h' + cloud + '-delta');
    if (dEl) {
      dEl.textContent = (isUp ? '▲ ' : '▼ ') + Math.abs(pct) + '%';
      dEl.className   = 'hcard-delta ' + (isUp ? 'up' : 'down');
    }
 
    // Prefer true historical stats from API over 30-point window
    const minV = meta?.min ?? sumRow?.min_price ?? v.min;
    const maxV = meta?.max ?? sumRow?.max_price ?? v.max;
    const avgV = meta?.avg ?? sumRow?.avg_price ?? v.avg;
    set('h' + cloud + '-min', fmt(minV));
    set('h' + cloud + '-max', fmt(maxV));
    set('h' + cloud + '-avg', fmt(avgV));
 
    const spread   = maxV - minV;
    const volLabel = spread < 0.015 ? 'Low' : spread < 0.035 ? 'Med' : 'High';
    const volCls   = spread < 0.015 ? 'vol-low' : spread < 0.035 ? 'vol-med' : 'vol-high';
    const vEl = document.getElementById('h' + cloud + '-vol');
    if (vEl) { vEl.textContent = volLabel; vEl.className = 'hcard-vol-val ' + volCls; }
 
    const badge = document.getElementById('best-' + cloud);
    if (badge) badge.style.display = cloud === bestCloud ? 'inline' : 'none';
  });
}

  // ── Price table ───────────────────────────────────────────────────
  function updatePriceTable(rows) {
    if (!rows || rows.length === 0) return;
    const tbody = document.querySelector('.price-table tbody');
    if (!tbody) return;
    const cls = p => parseFloat(p) < 0.25 ? 'low' : parseFloat(p) < 0.50 ? 'mid' : 'high';
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td><span class="cloud-tag ${r.cloud}">${r.cloud}</span></td>
        <td>${r.instance_type}</td>
        <td>${r.gpu_class}</td>
        <td>${r.availability_zone || r.region}</td>
        <td class="price-val ${cls(r.price_usd_per_hr)}">${parseFloat(r.price_usd_per_hr).toFixed(4)}</td>
        <td>${r.discount_pct != null ? r.discount_pct.toFixed(0) + '%' : '—'}</td>
      </tr>
    `).join('');
  }

  // ── Risk bars ─────────────────────────────────────────────────────
function updateRiskBars(riskData) {
  const panel = document.getElementById('risk-panel-body');
  if (!panel) return;
  if (!riskData || !Array.isArray(riskData)) return;  // ← add this line

  const badge = panel.closest('.panel')?.querySelector('.badge');

  if (riskData?.error) {
    if (badge) { badge.textContent = 'unavailable'; badge.className = 'badge badge-warn'; }
    panel.innerHTML = `<div style="font-family:var(--font-mono);font-size:11px;
      color:var(--muted);padding:16px 0;text-align:center">
      ${riskData.error}<br>
      <span style="opacity:.6">${riskData.retry_after_seconds
        ? 'Retrying in ' + riskData.retry_after_seconds + 's…'
        : 'Will retry on next refresh.'}</span>
    </div>`;
    return;
  }

  if (badge) {
    const now = new Date().toTimeString().slice(0, 5);
    badge.textContent = 'LSTM + XGBoost · ' + now;
    badge.className   = 'badge badge-live';
  }

  if (!riskData || riskData.length === 0) {
    panel.innerHTML = `<div style="font-family:var(--font-mono);font-size:11px;
      color:var(--muted);padding:16px 0;text-align:center">
      No instances to score.</div>`;
    return;
  }

  const bars = riskData.map(r => {
    const pct    = Math.round(r.risk * 100);
    const cls    = r.risk < 0.3 ? 'risk-low' : r.risk < 0.6 ? 'risk-med' : 'risk-high';
    const region = r.az || r.region || '';
    return `
      <div class="risk-item ${cls}" style="transition:opacity 0.3s">
        <div class="risk-header">
          <span class="risk-label">
            <span class="cloud-tag ${r.cloud}">${r.cloud}</span>
            ${r.instance_type} · ${region}
          </span>
          <span class="risk-score">${r.risk.toFixed(2)}</span>
        </div>
        <div class="risk-bar-bg">
          <div class="risk-bar-fill" style="width:0%;transition:width 0.6s ease"></div>
        </div>
      </div>`;
  }).join('');

  const worst = riskData[0];
  const best  = [...riskData].sort((a, b) => a.risk - b.risk)[0];
  const decision = worst.risk >= 0.6 ? `
    <div class="decision-box">
      <div class="decision-action migrate">MIGRATE</div>
      <div class="decision-reason">${worst.instance_type} risk > threshold (0.60)</div>
      <div class="decision-target">→ move to
        <span style="color:var(--accent2)">${best.cloud} ${best.instance_type}</span>
        · risk: ${best.risk.toFixed(2)}
      </div>
    </div>` : `
    <div class="decision-box">
      <div class="decision-action" style="background:var(--accent)">STABLE</div>
      <div class="decision-reason">All instances below migration threshold (0.60)</div>
    </div>`;

  panel.innerHTML = bars + decision;

  // Animate bars in after DOM paint
  requestAnimationFrame(() => {
    riskData.forEach((r, i) => {
      const fills = panel.querySelectorAll('.risk-bar-fill');
      if (fills[i]) fills[i].style.width = Math.round(r.risk * 100) + '%';
    });
  });
}

  // ── Event log ─────────────────────────────────────────────────────
  const LOG_TYPES = {
    'log-ok':   '[OK]',
    'log-warn': '[WARN]',
    'log-err':  '[ERR]',
    'log-info': '[INFO]',
  };

  function addLog(cls, msg) {
    const feed = document.getElementById('log-feed');
    if (!feed) return;
    const now   = new Date().toTimeString().slice(0, 8);
    const label = LOG_TYPES[cls] || '[INFO]';
    const div   = document.createElement('div');
    div.className = 'log-entry';
    div.innerHTML = `<span class="log-time">${now}</span><span class="${cls}">${label}</span><span class="log-msg"> ${msg}</span>`;
    feed.insertBefore(div, feed.firstChild);
    if (feed.children.length > 25) feed.removeChild(feed.lastChild);
  }

  function addPreemptionLogs(events) {
    if (!events || events.length === 0) return;
    events.slice(0, 3).forEach(e => {
      addLog('log-err',
        `Preemption · ${e.cloud} ${e.instance_type} · ${e.region} · src: ${e.preemption_source || '?'}`
      );
    });
  }

  // ── Stats (total rows / preemptions) ──────────────────────────────
  /**
   * Writes to BOTH locations:
   *   stat-ckpt-total / stat-ckpt-emergency  → top stat card
   *   ckpt-total / ckpt-emergency            → jobs panel footer
   *
   * The old code only wrote to ckpt-total/ckpt-emergency which
   * are duplicated IDs — getElementById only found the first match
   * (the jobs panel), so the stat card never updated.
   */
  // ui.js — replace updateStats()
function updateStats(data) {
    if (!data) return;

    const set = (id, v) => {
        const el = document.getElementById(id);
        if (el) el.textContent = v;
    };

    if (data.error) {
        addLog('log-warn', `Stats BQ error: ${data.error.slice(0, 80)}`);
    }

    const total       = data.total_rows       != null ? data.total_rows.toLocaleString() : '—';
    const preemptions = data.preemption_count != null ? data.preemption_count             : '—';

    set('ckpt-total',     total);
    set('ckpt-emergency', preemptions);
}

  // ── Jobs list ─────────────────────────────────────────────────────
  let _jobs       = [];
  let _jobsLoaded = false;

  const ACTIVE_STATUSES   = new Set(['running', 'migrating', 'queued']);
  const POLLABLE_STATUSES = new Set(['running', 'migrating', 'queued', 'preempted']);

  function renderJobs() {
    const el = document.getElementById('jobs-list');
    if (!el) return;

    if (!_jobsLoaded) {
      el.innerHTML = `<div style="font-family:var(--font-mono);font-size:11px;
        color:var(--muted);padding:16px 0;text-align:center">
        Loading jobs from S3…</div>`;
      const badge = document.getElementById('job-count-badge');
      if (badge) badge.textContent = '…';
      return;
    }

    if (_jobs.length === 0) {
      el.innerHTML = `<div style="font-family:var(--font-mono);font-size:11px;
        color:var(--muted);padding:16px 0;text-align:center">
        No jobs yet — click + Add Job to submit one.</div>`;
      const badge = document.getElementById('job-count-badge');
      if (badge) badge.textContent = '0 active';
      return;
    }

    const statusClass = s => ({
      running:         's-running',
      migrating:       's-migrating',
      paused:          's-paused',
      done:            's-done',
      queued:          's-paused',
      preempted:       's-migrating',
      budget_exceeded: 's-done',
      launch_failed:   's-paused',
    }[s] || 's-paused');

    el.innerHTML = _jobs.map((j, i) => `
      <div class="job-row" id="jobrow-${j.id}">
        <div class="job-id">${j.id.slice(0, 14)}</div>
        <div class="job-info">
          <div class="job-name">${j.name}</div>
          <div class="job-meta">${j.meta}</div>
        </div>
        <div class="job-prog">
          <div class="job-prog-bar">
            <div class="job-prog-fill" style="width:${j.pct}%"></div>
          </div>
          <div class="job-pct">${j.pct}%</div>
        </div>
        <div class="job-status ${statusClass(j.status)}">${j.status}</div>
        <div class="job-actions">
          ${j.console_url
            ? `<button class="btn-sm" onclick="window.open('${j.console_url}','_blank')">logs</button>`
            : ''}
          <button class="btn-sm" onclick="UI.pauseJob(${i})">${j.status === 'paused' ? 'resume' : 'pause'}</button>
          <button class="btn-sm danger" onclick="UI.removeJob(${i})">✕</button>
        </div>
      </div>
    `).join('');

    const active = _jobs.filter(j => ACTIVE_STATUSES.has(j.status)).length;
    const badge  = document.getElementById('job-count-badge');
    if (badge) badge.textContent = active + ' active';
  }

  function pauseJob(i) {
    if (!_jobs[i]) return;
    const job  = _jobs[i];
    job.status = job.status === 'paused' ? 'running' : 'paused';
    addLog('log-warn', `Job ${job.id} ${job.status}`);
    renderJobs();
  }

  function removeJob(i) {
    if (!_jobs[i]) return;
    addLog('log-err', `Job ${_jobs[i].id} removed from dashboard`);
    _jobs.splice(i, 1);
    renderJobs();
  }

  function addJob(job) {
    const existing = _jobs.findIndex(j => j.id === job.id);
    if (existing >= 0) {
      _jobs[existing] = { ..._jobs[existing], ...job };
    } else {
      _jobs.push(job);
    }
    renderJobs();
    addLog('log-ok', `Job ${job.id} added to dashboard`);
  }

  function updateJobFromState(state) {
    const i = _jobs.findIndex(j => j.id === state.job_id);
    if (i < 0) return;

    const epoch  = state.epoch        ?? 0;
    const total  = state.total_epochs ?? 50;
    const pct    = Math.min(99, Math.round((epoch / Math.max(total, 1)) * 100));

    const inst    = state.instance_type || '';
    const cloud   = state.cloud         || '';
    const costStr = state.cost_usd != null ? ` · $${parseFloat(state.cost_usd).toFixed(3)}` : '';
    const lossStr = state.loss     != null ? ` · loss ${parseFloat(state.loss).toFixed(4)}`  : '';

    _jobs[i].status = state.status || _jobs[i].status;
    _jobs[i].pct    = pct;
    _jobs[i].meta   = `${cloud} ${inst} · epoch ${epoch}/${total}${lossStr}${costStr}`.trim();

    renderJobs();
  }

  function mergeRealJobs(realJobs) {
    _jobsLoaded = true;

    if (!realJobs || realJobs.length === 0) {
      renderJobs();
      return;
    }

    const gcsIds    = new Set(realJobs.map(j => j.job_id).filter(Boolean));
    const localOnly = _jobs.filter(j =>
      j.id && j.id !== 'undefined' &&
      !gcsIds.has(j.id) && !j.id.startsWith('demo-')
    );

    const statusMap = {
      running:         'running',
      queued:          'running',
      launched:        'running',
      migrating:       'migrating',
      paused:          'paused',
      done:            'done',
      preempted:       'preempted',
      budget_exceeded: 'done',
      launch_failed:   'paused',
      failed:          'paused',
    };

    const fromS3 = realJobs
      .filter(j => j.job_id)
      .map(j => ({
        id:          j.job_id,
        name:        j.task_name || j.job_id,
        meta:        _buildMeta(j),
        pct:         j.progress_pct ??
                     Math.min(99, Math.round(((j.epoch || 0) / Math.max(j.total_epochs || 50, 1)) * 100)),
        status:      statusMap[j.status] || 'running',
        cloud:       j.cloud || '',
        console_url: j.console_url || '',
      }));

    _jobs = [...fromS3, ...localOnly];
    renderJobs();
  }

  // ui.js — pollActiveJobs()
function pollActiveJobs() {
    const targets = _jobs.filter(j =>
        j.id &&
        j.id !== 'undefined' &&
        !j.id.startsWith('demo-') &&
        POLLABLE_STATUSES.has(j.status)
    );
    targets.forEach(async job => {
        try {
            const resp = await fetch(`http://localhost:5050/api/jobs/${job.id}`, {
                signal: AbortSignal.timeout(15_000),   // ← was 5000
            });
            if (!resp.ok) return;
            const state = await resp.json();
            if (state && !state.error) updateJobFromState(state);
        } catch (e) {
            if (e.name !== 'TimeoutError' && e.name !== 'AbortError') {
                console.warn('[pollActiveJobs]', job.id, e.message);
            }
        }
    });
}

  function _buildMeta(state) {
    const parts = [];
    const inst  = state.instance_type
               || state.launch_result?.instance_type
               || '';
    const cloud = state.cloud || state.launch_result?.cloud || '';

    if (inst || cloud)           parts.push(`${cloud} ${inst}`.trim());
    if (state.epoch != null)     parts.push(`epoch ${state.epoch}/${state.total_epochs ?? '?'}`);
    if (state.step)              parts.push(`step ${state.step}`);
    if (state.loss     != null)  parts.push(`loss ${parseFloat(state.loss).toFixed(4)}`);
    if (state.accuracy != null)  parts.push(`acc ${parseFloat(state.accuracy).toFixed(3)}`);
    if (state.cost_usd != null)  parts.push(`$${parseFloat(state.cost_usd).toFixed(3)}`);

    if (state.status === 'done')            parts.push('✓ complete');
    if (state.status === 'preempted')       parts.push('⚡ preempted');
    if (state.status === 'queued')          parts.push('⏳ VM booting…');
    if (state.status === 'launched')        parts.push('⏳ VM booting…');
    if (state.status === 'budget_exceeded') parts.push('⚠ budget limit');
    if (state.status === 'failed')          parts.push('✗ failed');

    return parts.join(' · ') || state.status || '—';
  }

  return {
    startClock,
    updateStatCards,
    updateHealthStrip,
    updatePriceTable,
    updateRiskBars,
    addLog, addPreemptionLogs,
    updateStats,
    renderJobs, pauseJob, removeJob, addJob,
    mergeRealJobs, pollActiveJobs, updateJobFromState,
  };
})();