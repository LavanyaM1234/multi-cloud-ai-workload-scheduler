/**
 * static/js/ui.js  — updated
 * ────────────────────────────
 * Key changes:
 *   - addJob() accepts console_url and renders a link in the job row
 *   - pollActiveJobs() added — calls /api/jobs/status/<id> every 10s
 *     for running jobs to update progress without waiting for the 60s refresh
 *   - job progress now reads epoch/total_epochs from real job_state.json
 */

const UI = (() => {

  function startClock() {
    const tick = () => {
      const el = document.getElementById('clock');
      if (el) el.textContent = new Date().toUTCString().slice(17, 25) + ' UTC';
    };
    tick();
    setInterval(tick, 1000);
  }

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
        dEl.className = 'stat-delta ' + (pct >= 0 ? 'delta-up' : 'delta-down');
      }
    });
  }

  function updateNumericCards(summary) {
    if (!summary) return;
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const d = summary[cloud]; if (!d) return;
      const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = '$' + parseFloat(val).toFixed(4);
      };
      set(cloud + '-current', d.current_price);
      set(cloud + '-min',     d.min_price);
      set(cloud + '-max',     d.max_price);
      set(cloud + '-avg',     d.avg_price);
    });
  }

  function updateNumericCardsMock(values) {
    ['aws', 'gcp', 'azure'].forEach(cloud => {
      const v = values[cloud]; if (!v) return;
      const fmt = n => '$' + n.toFixed(4);
      const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      set(cloud + '-current', fmt(v.cur));
      set(cloud + '-min',     fmt(v.min));
      set(cloud + '-max',     fmt(v.max));
      set(cloud + '-avg',     fmt(v.avg));
      const pEl = document.getElementById(cloud + '-price');
      const dEl = document.getElementById(cloud + '-delta');
      if (pEl) pEl.textContent = '$' + v.cur.toFixed(3);
      if (dEl) {
        const pct = ((v.cur - v.prev) / v.prev * 100).toFixed(1);
        dEl.textContent = (pct >= 0 ? '▲ ' : '▼ ') + Math.abs(pct) + '% from last poll';
        dEl.className = 'stat-delta ' + (pct >= 0 ? 'delta-up' : 'delta-down');
      }
    });
  }

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

  const LOG_TYPES = { 'log-ok': '[OK]', 'log-warn': '[WARN]', 'log-err': '[ERR]', 'log-info': '[INFO]' };

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

  function updateStats(data) {
    if (!data) return;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('ckpt-total',     data.total_rows       || '—');
    set('ckpt-emergency', data.preemption_count || '0');
  }

  // ── Jobs ───────────────────────────────────────────────────────────────────

  let _jobs = [
    { id: 'demo-job-001', name: 'MLP · lr=0.001 · h=256',
      meta: 'AWS g4dn.xlarge · step 3200', pct: 33, status: 'running', cloud: 'aws', console_url: '' },
    { id: 'demo-job-002', name: 'MLP · lr=0.0005 · h=128',
      meta: 'Migrating AWS → GCP · last ckpt step 1850', pct: 19, status: 'migrating', cloud: 'gcp', console_url: '' },
  ];

  function renderJobs() {
    const el = document.getElementById('jobs-list');
    if (!el) return;
    const statusClass = s => ({
      running: 's-running', migrating: 's-migrating', paused: 's-paused',
      done: 's-done', queued: 's-paused', preempted: 's-migrating',
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

    const active = _jobs.filter(j => ['running', 'migrating', 'queued'].includes(j.status)).length;
    const badge  = document.getElementById('job-count-badge');
    if (badge) badge.textContent = active + ' active';
  }

  function pauseJob(i) {
    if (!_jobs[i]) return;
    _jobs[i].status = _jobs[i].status === 'paused' ? 'running' : 'paused';
    _jobs[i].meta   = _jobs[i].status === 'paused' ? 'Manually paused' : 'Resumed';
    addLog('log-warn', `Job ${_jobs[i].id} ${_jobs[i].status}`);
    renderJobs();
  }

  function removeJob(i) {
    if (!_jobs[i]) return;
    addLog('log-err', `Job ${_jobs[i].id} removed from dashboard`);
    _jobs.splice(i, 1);
    renderJobs();
  }

  function addJob(job) {
    // Avoid duplicates
    const existing = _jobs.findIndex(j => j.id === job.id);
    if (existing >= 0) { _jobs[existing] = { ..._jobs[existing], ...job }; }
    else { _jobs.push(job); }
    renderJobs();
    addLog('log-ok', `Job ${job.id} added to dashboard`);
  }

  /**
   * Update a specific job row with fresh data from job_state.json.
   * Called by pollActiveJobs() every 10s.
   */
  function updateJobFromState(state) {
    const i = _jobs.findIndex(j => j.id === state.job_id);
    if (i < 0) return;

    const pct = state.total_epochs
      ? Math.round((state.epoch / state.total_epochs) * 100)
      : _jobs[i].pct;

    const costStr = state.cost_usd != null ? ` · $${state.cost_usd.toFixed(3)}` : '';
    const lossStr = state.loss     != null ? ` · loss ${state.loss.toFixed(4)}` : '';

    _jobs[i].status = state.status || _jobs[i].status;
    _jobs[i].pct    = Math.min(pct, 99);
    _jobs[i].meta   = `${state.cloud || 'gcp'} ${state.instance || ''} · ` +
                      `epoch ${state.epoch}/${state.total_epochs}` + lossStr + costStr;
    renderJobs();
  }

 function mergeRealJobs(realJobs) {
    if (!realJobs || realJobs.length === 0) return;
 
    // Keep any locally-submitted jobs that aren't in GCS yet
    const gcsIds = new Set(realJobs.map(j => j.job_id));
    const localOnly = _jobs.filter(j => !gcsIds.has(j.id));
 
    const statusMap = {
      running:        'running',
      queued:         'running',    // show as running while VM boots
      migrating:      'migrating',
      paused:         'paused',
      done:           'done',
      preempted:      'paused',
      budget_exceeded:'done',
      launch_failed:  'paused',
    };
 
    const fromGcs = realJobs.map(j => ({
      id:     j.job_id,
      name:   j.task_name || j.job_id,
      meta:   _buildMeta(j),
      pct:    j.progress_pct ?? Math.min(99, Math.round(((j.epoch || 0) / 50) * 100)),
      status: statusMap[j.status] || 'running',
      cloud:  j.cloud || 'gcp',
    }));
 
    _jobs = [...fromGcs, ...localOnly];
    renderJobs();
  }
 

  /**
   * Poll active jobs every 10s to get live progress.
   * Calls /api/jobs/status/<job_id> for each running job.
   */
  function pollActiveJobs() {
    const active = _jobs.filter(j => ['running', 'migrating', 'queued'].includes(j.status));
    active.forEach(async job => {
      try {
        const resp = await fetch(`/api/jobs/status/${job.id}`,
          { signal: AbortSignal.timeout(5000) });
        if (!resp.ok) return;
        const state = await resp.json();
        if (state && !state.error) updateJobFromState(state);
      } catch (e) {
        // silently ignore — job may not have written state yet
      }
    });
  }

   function _buildMeta(state) {
    const parts = [];
    if (state.instance)  parts.push(`GCP ${state.instance}`);
    if (state.epoch)     parts.push(`epoch ${state.epoch}`);
    if (state.step)      parts.push(`step ${state.step}`);
    if (state.loss != null) parts.push(`loss ${parseFloat(state.loss).toFixed(4)}`);
    if (state.accuracy != null) parts.push(`acc ${parseFloat(state.accuracy).toFixed(3)}`);
    if (state.cost_usd != null) parts.push(`$${parseFloat(state.cost_usd).toFixed(3)}`);
    if (state.status === 'done') parts.push('✓ complete');
    if (state.status === 'preempted') parts.push('⚡ preempted');
    if (state.status === 'queued') parts.push('⏳ VM booting...');
    return parts.join(' · ') || state.status;
  }
 

  return {
    startClock,
    updateStatCards, updateNumericCards, updateNumericCardsMock,
    updatePriceTable, addLog, addPreemptionLogs, updateStats,
    renderJobs, pauseJob, removeJob, addJob,
    mergeRealJobs, pollActiveJobs, updateJobFromState,
  };
})();