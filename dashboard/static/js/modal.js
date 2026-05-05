/**
 * static/js/modal.js
 * ───────────────────
 * Multi-step job submission modal.
 * Steps: Task → Dataset → Model → Compute → Submit
 */

const MODAL = (() => {
  let _step = 1;
  const TOTAL = 4;

  const DATASETS = {
    'synthetic-500k': { rows: '500,000', features: '50', classes: '5', task: 'classif.', path: 's3://ml-scheduler-dataset/synthetic/train/' },
    'synthetic-100k': { rows: '100,000', features: '50', classes: '5', task: 'classif.', path: 's3://ml-scheduler-dataset/synthetic/train/shard_000*.csv' },
    'custom':         { rows: '—',       features: '—',  classes: '—', task: '—',        path: '' },
  };

  function open() {
    _step = 1;
    _goStep(1);
    document.getElementById('modal').classList.add('open');

    // Priority option click handlers
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
    _goStep(1);
  }

  function _goStep(n) {
    for (let i = 1; i <= TOTAL; i++) {
      const panel = document.getElementById('step-' + i);
      const tab   = document.getElementById('tab-' + i);
      if (panel) panel.style.display = i === n ? 'block' : 'none';
      if (tab)   tab.classList.toggle('active', i === n);
    }
    _step = n;
    const btnBack = document.getElementById('btn-back');
    const btnNext = document.getElementById('btn-next');
    if (btnBack) btnBack.style.display = n === 1 ? 'none' : 'block';
    if (btnNext) btnNext.textContent   = n === TOTAL ? 'Submit Job' : 'Next →';
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
      if (inp) inp.value = d.features !== '—' ? d.features.replace(',', '') : '50';
      if (cls) cls.value = d.classes  !== '—' ? d.classes : '5';
    }
  }

  function _val(id) { const el = document.getElementById(id); return el ? el.value : ''; }

  function _updateSummary() {
    const taskname = _val('f-taskname') || 'Untitled job';
    const jobId    = 'job-' + new Date().toISOString().slice(5, 10).replace('-', '') + '-' + Math.random().toString(36).slice(2, 8);
    const items    = [
      ['Job ID',   jobId,                       ''],
      ['Task',     taskname,                    ''],
      ['Dataset',  _val('f-dataset'),            ''],
      ['Priority', document.querySelector('input[name="priority"]:checked')?.value || 'balanced', ''],
      ['Config',   `lr=${_val('f-lr')} · h=${_val('f-hidden')} · e=${_val('f-epochs')}`,          ''],
      ['Compute',  `${_val('f-cloud')} · ${_val('f-instance')}`, ''],
      ['Budget',   '$' + _val('f-budget'),       'color:var(--accent2)'],
      ['Fallback', _val('f-fallback'),            ''],
    ];
    const box = document.getElementById('summary-content');
    if (!box) return;
    box.innerHTML = items.map(([label, val, style]) =>
      `<div><span style="color:var(--muted)">${label}</span><br><span style="${style}">${val}</span></div>`
    ).join('');
    box.dataset.jobId = jobId;
  }

  async function _submit() {
    const summary = document.getElementById('summary-content');
    const jobId   = summary?.dataset.jobId || 'job-new';
    const body    = {
      job_id:       jobId,
      model_arch:   'mlp',
      dataset_size: parseFloat(_val('f-inputdim')) || 10,
      max_budget:   parseFloat(_val('f-budget'))   || 5,
      deadline_hrs: parseFloat(_val('f-maxhrs'))   || 8,
      min_gpu_mem:  8,
    };

    // Call scheduler API
    const decision = await API.submitJob(body);

    UI.addJob({
      id:     jobId,
      name:   `${_val('f-taskname') || 'Untitled'} · lr=${_val('f-lr')} · h=${_val('f-hidden')}`,
      meta:   decision?.error
        ? `Queued (scheduler offline) · ${_val('f-instance')}`
        : `${decision.cloud?.toUpperCase()} ${decision.instance_type} · est $${decision.est_cost?.toFixed(2)} · ${decision.est_hours?.toFixed(1)}h`,
      pct:    0,
      status: 'running',
      cloud:  decision?.cloud || 'aws',
    });

    close();
  }

  return { open, close, next, prev, updateDatasetMeta };
})();
