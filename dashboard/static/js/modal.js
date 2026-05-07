/**
 * static/js/modal.js  — updated
 * ──────────────────────────────
 * Key change: _submit() now sends ALL form fields to /api/jobs/submit
 * and handles the real response (job_id, console_url, launched: true/false).
 *
 * After a successful launch, the job appears in the jobs panel immediately
 * with status "queued" and updates to "running" once the VM boots and
 * writes its first job_state.json (~2 minutes after launch).
 */

const MODAL = (() => {
  let _step = 1;
  const TOTAL = 4;

  const DATASETS = {
    'synthetic-500k': {
      rows: '500,000', features: '50', classes: '5', task: 'classif.',
      path: 's3://ml-scheduler-dataset/synthetic/train/',
    },
    'synthetic-100k': {
      rows: '100,000', features: '50', classes: '5', task: 'classif.',
      path: 's3://ml-scheduler-dataset/synthetic/train/shard_000*.csv',
    },
    'custom': { rows: '—', features: '—', classes: '—', task: '—', path: '' },
  };

  function open() {
    _step = 1;
    _goStep(1);
    document.getElementById('modal').classList.add('open');
    ['cheapest', 'balanced', 'fastest'].forEach(p => {
      const el = document.getElementById('pri-' + p);
      if (el) el.onclick = () => {
        document.querySelectorAll('.priority-opt').forEach(x => x.classList.remove('selected'));
        el.classList.add('selected');
        el.querySelector('input').checked = true;
      };
    });
  }

  function close() {
    document.getElementById('modal').classList.remove('open');
  }

  function _goStep(n) {
    for (let i = 1; i <= TOTAL; i++) {
      const panel = document.getElementById('step-' + i);
      const tab   = document.getElementById('tab-' + i);
      if (panel) panel.style.display = (i === n) ? 'block' : 'none';
      if (tab)   tab.classList.toggle('active', i === n);
    }
    _step = n;
    const btnBack = document.getElementById('btn-back');
    const btnNext = document.getElementById('btn-next');
    if (btnBack) btnBack.style.display = (n === 1) ? 'none' : 'block';
    if (btnNext) btnNext.textContent   = (n === TOTAL) ? 'Launch Job' : 'Next →';
    if (n === TOTAL) _updateSummary();
  }

  function next() { if (_step < TOTAL) _goStep(_step + 1); else _submit(); }
  function prev() { if (_step > 1)     _goStep(_step - 1); }

  function updateDatasetMeta() {
    const val = document.getElementById('f-dataset')?.value;
    if (!val) return;
    const d = DATASETS[val] || DATASETS['synthetic-500k'];
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('ds-rows', d.rows); set('ds-features', d.features);
    set('ds-classes', d.classes); set('ds-task', d.task);
    const path = document.getElementById('ds-path');
    if (path) path.textContent = d.path;
    const metaWrap   = document.getElementById('dataset-meta-wrap');
    const customWrap = document.getElementById('custom-s3-wrap');
    if (metaWrap)   metaWrap.style.display   = val === 'custom' ? 'none'  : 'block';
    if (customWrap) customWrap.style.display  = val === 'custom' ? 'block' : 'none';
    if (val !== 'custom') {
      const inp = document.getElementById('f-inputdim');
      const cls = document.getElementById('f-numclasses');
      if (inp) inp.value = d.features !== '—' ? d.features.replace(',','') : '50';
      if (cls) cls.value = d.classes  !== '—' ? d.classes                  : '5';
    }
  }

  function _val(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
  }

  function _updateSummary() {
    const jobId = 'job-' + new Date().toISOString().slice(5,10).replace('-','') +
                  '-' + Math.random().toString(36).slice(2,8);
    const items = [
      ['Job ID',    jobId,                                  'color:var(--accent)'],
      ['Task',      _val('f-taskname') || 'Untitled',       ''],
      ['Dataset',   _val('f-dataset'),                      ''],
      ['Priority',  document.querySelector('input[name="priority"]:checked')?.value || 'balanced', ''],
      ['Config',    `lr=${_val('f-lr')} h=${_val('f-hidden')} e=${_val('f-epochs')}`, ''],
      ['Compute',   `${_val('f-cloud')} · ${_val('f-instance')}`,                    ''],
      ['Budget',    '$' + _val('f-budget'),                 'color:var(--accent2)'],
      ['Fallback',  _val('f-fallback'),                     ''],
    ];
    const box = document.getElementById('summary-content');
    if (!box) return;
    box.innerHTML = items.map(([label, val, style]) =>
      `<div>
        <span style="color:var(--muted)">${label}</span><br>
        <span style="${style}">${val}</span>
      </div>`
    ).join('');
    box.dataset.jobId = jobId;
  }

  async function _submit() {
    const btnNext = document.getElementById('btn-next');
    if (btnNext) {
      btnNext.textContent = 'Launching...';
      btnNext.disabled    = true;
    }

    // ── Collect ALL form fields ──────────────────────────────────
    const body = {
      // Step 1 — Task
      task_name:    _val('f-taskname') || 'Untitled',
      priority:     document.querySelector('input[name="priority"]:checked')?.value || 'balanced',
      max_budget:   parseFloat(_val('f-budget'))   || 2.0,
      deadline_hrs: parseFloat(_val('f-maxhrs'))   || 8.0,

      // Step 2 — Dataset
      dataset:      _val('f-dataset'),
      s3_path:      _val('f-s3path'),      // only used if dataset = 'custom'
      input_dim:    parseInt(_val('f-inputdim'))    || 50,
      num_classes:  parseInt(_val('f-numclasses'))  || 5,

      // Step 3 — Model
      lr:           parseFloat(_val('f-lr'))        || 0.001,
      hidden_dim:   parseInt(_val('f-hidden'))       || 256,
      dropout:      parseFloat(_val('f-dropout'))   || 0.3,
      batch_size:   parseInt(_val('f-batch'))        || 64,
      epochs:       parseInt(_val('f-epochs'))       || 50,
      ckpt_every:   parseInt(_val('f-ckpt'))         || 50,

      // Step 4 — Compute
      cloud:        _val('f-cloud'),        // 'auto' | 'aws' | 'gcp' | 'azure'
      instance:     _val('f-instance'),
      fallback:     _val('f-fallback'),
    };

    try {
      const resp = await API.submitJob(body);

      if (resp.error) {
        UI.addLog('log-err', `Launch failed: ${resp.error}`);
        alert(`Launch failed: ${resp.error}`);
        return;
      }

      // ── Job launched — add to dashboard immediately ───────────
      UI.addJob({
        id:     resp.job_id,
        name:   `${body.task_name} · lr=${body.lr} · h=${body.hidden_dim}`,
        meta:   `${resp.cloud?.toUpperCase()} ${resp.instance_type} · ` +
                `$${resp.price_usd_hr?.toFixed(3)}/hr · queued (VM booting ~2min)`,
        pct:    0,
        status: 'running',
        cloud:  resp.cloud || 'gcp',
        console_url: resp.console_url || '',
      });

      UI.addLog('log-ok',
        `Job ${resp.job_id} launched → ${resp.cloud} ${resp.instance_type} ` +
        `· est $${resp.est_cost?.toFixed(2)}`
      );

      if (resp.console_url) {
        UI.addLog('log-info',
          `VM console: <a href="${resp.console_url}" target="_blank" ` +
          `style="color:var(--accent)">open</a>`
        );
      }

      close();

    } catch (e) {
      UI.addLog('log-err', `Submit error: ${e.message}`);
      alert(`Error: ${e.message}`);
    } finally {
      if (btnNext) {
        btnNext.textContent = 'Launch Job';
        btnNext.disabled    = false;
      }
    }
  }

  // expose _goStep for tab onclick handlers in HTML
  return { open, close, next, prev, updateDatasetMeta, _goStep };
})();