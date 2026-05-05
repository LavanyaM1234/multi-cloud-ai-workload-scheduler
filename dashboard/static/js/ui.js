/**
 * static/js/ui.js
 * ────────────────
 * All DOM manipulation — updates stat cards, price table,
 * job list, event log, and numeric cards.
 *
 * ui.js knows nothing about fetch or canvas.
 * chart.js knows nothing about the DOM.
 * api.js knows nothing about the DOM or canvas.
 *
 * main.js wires all three together.
 */

const UI = (() => {

  // ── Clock ─────────────────────────────────────────────────────────
  function startClock() {
    function tick() {
      const el = document.getElementById('clock');
      if (el) el.textContent = new Date().toUTCString().slice(17, 25) + ' UTC';
    }
    tick();
    setInterval(tick, 1000);
  }

  // ── Stat cards (top row) ──────────────────────────────────────────
  function updateStatCards(summary) {
    if (!summary) return;
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const d   = summary[cloud];
      if (!d) return;
      const cur  = parseFloat(d.current_price);
      const avg  = parseFloat(d.avg_price);
      const pct  = ((cur - avg) / avg * 100).toFixed(1);
      const pEl  = document.getElementById(cloud + '-price');
      const dEl  = document.getElementById(cloud + '-delta');
      if (pEl) pEl.textContent = '$' + cur.toFixed(3);
      if (dEl) {
        dEl.textContent = (pct >= 0 ? '▲ ' : '▼ ') + Math.abs(pct) + '% vs 3hr avg';
        dEl.className   = 'stat-delta ' + (pct >= 0 ? 'delta-up' : 'delta-down');
      }
    });
  }

  // ── Numeric cards (below chart) ───────────────────────────────────
  function updateNumericCards(summary) {
    if (!summary) return;
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const d = summary[cloud];
      if (!d) return;
      const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = '$' + parseFloat(val).toFixed(4); };
      set(cloud + '-current', d.current_price);
      set(cloud + '-min',     d.min_price);
      set(cloud + '-max',     d.max_price);
      set(cloud + '-avg',     d.avg_price);
    });
  }

  // ── Numeric cards from mock chart values ──────────────────────────
  function updateNumericCardsMock(values) {
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const v   = values[cloud];
      const fmt = n => '$' + n.toFixed(4);
      const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      set(cloud + '-current', fmt(v.cur));
      set(cloud + '-min',     fmt(v.min));
      set(cloud + '-max',     fmt(v.max));
      set(cloud + '-avg',     fmt(v.avg));
      // also update stat card
      const pEl = document.getElementById(cloud + '-price');
      const dEl = document.getElementById(cloud + '-delta');
      if (pEl) pEl.textContent = '$' + v.cur.toFixed(3);
      if (dEl) {
        const pct = ((v.cur - v.prev) / v.prev * 100).toFixed(1);
        dEl.textContent = (pct >= 0 ? '▲ ' : '▼ ') + Math.abs(pct) + '% from last poll';
        dEl.className   = 'stat-delta ' + (pct >= 0 ? 'delta-up' : 'delta-down');
      }
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
        <td>${r.discount_pct ? r.discount_pct.toFixed(0) + '%' : '—'}</td>
      </tr>
    `).join('');
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
    div.className   = 'log-entry';
    div.innerHTML   = `<span class="log-time">${now}</span><span class="${cls}">${label}</span><span class="log-msg"> ${msg}</span>`;
    feed.insertBefore(div, feed.firstChild);
    if (feed.children.length > 20) feed.removeChild(feed.lastChild);
  }

  function addPreemptionLogs(events) {
    if (!events || events.length === 0) return;
    events.slice(0, 3).forEach(e => {
      addLog('log-err',
        `Preemption · ${e.cloud} ${e.instance_type} · ${e.region} · source: ${e.preemption_source || '?'}`
      );
    });
  }

  // ── Stats (total rows, preemptions count) ─────────────────────────
  function updateStats(data) {
    if (!data) return;
    const el = id => document.getElementById(id);
    if (el('ckpt-total'))     el('ckpt-total').textContent     = data.total_rows || '—';
    if (el('ckpt-emergency')) el('ckpt-emergency').textContent = data.preemption_count || '0';
  }

  // ── Jobs list ─────────────────────────────────────────────────────
  let _jobs = [
    { id: 'demo-job-001', name: 'MLP · lr=0.001 · hidden=256', meta: 'AWS g4dn.xlarge · step 3200',                pct: 33, status: 'running',   cloud: 'aws' },
    { id: 'demo-job-002', name: 'MLP · lr=0.0005 · hidden=128', meta: 'Migrating AWS → GCP · last ckpt: step 1850', pct: 19, status: 'migrating', cloud: 'gcp' },
    { id: 'demo-job-003', name: 'MLP · lr=0.01 · dropout=0.5',  meta: 'Paused at step 720',                         pct:  7, status: 'paused',    cloud: 'aws' },
  ];

  function renderJobs() {
    const el = document.getElementById('jobs-list');
    if (!el) return;
    const statusClass = s => ({ running: 's-running', migrating: 's-migrating', paused: 's-paused', done: 's-done' }[s] || 's-paused');
    el.innerHTML = _jobs.map((j, i) => `
      <div class="job-row">
        <div class="job-id">${j.id}</div>
        <div class="job-info">
          <div class="job-name">${j.name}</div>
          <div class="job-meta">${j.meta}</div>
        </div>
        <div class="job-prog">
          <div class="job-prog-bar"><div class="job-prog-fill" style="width:${j.pct}%"></div></div>
          <div class="job-pct">${j.pct}%</div>
        </div>
        <div class="job-status ${statusClass(j.status)}">${j.status}</div>
        <div class="job-actions">
          <button class="btn-sm" onclick="UI.pauseJob(${i})">${j.status === 'paused' ? 'resume' : 'pause'}</button>
          <button class="btn-sm danger" onclick="UI.removeJob(${i})">✕</button>
        </div>
      </div>
    `).join('');
    const active = _jobs.filter(j => j.status === 'running' || j.status === 'migrating').length;
    const badge  = document.getElementById('job-count-badge');
    if (badge) badge.textContent = active + ' active';
  }

  function pauseJob(i) {
    _jobs[i].status = _jobs[i].status === 'paused' ? 'running' : 'paused';
    _jobs[i].meta   = _jobs[i].status === 'paused' ? 'Manually paused' : 'Resumed manually';
    addLog('log-warn', `Job ${_jobs[i].id} ${_jobs[i].status}`);
    renderJobs();
  }

  function removeJob(i) {
    addLog('log-err', `Job ${_jobs[i].id} removed`);
    _jobs.splice(i, 1);
    renderJobs();
  }

  function addJob(job) {
    _jobs.push(job);
    renderJobs();
    addLog('log-ok', `Job ${job.id} submitted`);
  }

  function mergeRealJobs(realJobs) {
    if (!realJobs || realJobs.length === 0) return;
    // Replace mock jobs with real ones from GCS
    _jobs = realJobs.map(j => ({
      id:     j.job_id,
      name:   j.job_id,
      meta:   `epoch ${j.epoch} · step ${j.step} · loss ${j.loss?.toFixed(4) || '?'}`,
      pct:    Math.min(99, Math.round((j.step / 10000) * 100)),
      status: j.status || 'running',
      cloud:  'gcp',
    }));
    renderJobs();
  }

  return {
    startClock,
    updateStatCards, updateNumericCards, updateNumericCardsMock,
    updatePriceTable, addLog, addPreemptionLogs,
    updateStats, renderJobs, pauseJob, removeJob, addJob,
    mergeRealJobs,
  };
})();
