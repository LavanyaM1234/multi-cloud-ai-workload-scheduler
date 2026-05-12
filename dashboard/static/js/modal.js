/**
 * static/js/modal.js
 */

const MODAL = (() => {
  let _step = 1;
  const TOTAL = 4;

  const DATASETS = {
    'synthetic-500k': {
      rows: '500,000', features: '50', classes: '5', task: 'classif.',
      path: '',   // no S3 path — generated on VM
    },
    'synthetic-100k': {
      rows: '100,000', features: '50', classes: '5', task: 'classif.',
      path: '',
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

    const metaWrap   = document.getElementById('dataset-meta-wrap');
    const customWrap = document.getElementById('custom-s3-wrap');
    if (metaWrap)   metaWrap.style.display  = val === 'custom' ? 'none'  : 'block';
    if (customWrap) customWrap.style.display = val === 'custom' ? 'block' : 'none';

    if (val !== 'custom') {
      const inp = document.getElementById('f-inputdim');
      const cls = document.getElementById('f-numclasses');
      if (inp) inp.value = d.features !== '—' ? d.features.replace(',', '') : '50';
      if (cls) cls.value = d.classes  !== '—' ? d.classes                   : '5';
    }
  }

  function _val(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
  }

  function _updateSummary() {
    const jobId = 'job-' + new Date().toISOString().slice(5, 10).replace('-', '') +
                  '-' + Math.random().toString(36).slice(2, 8);

    // ── Dataset display ───────────────────────────────────────────
    const datasetType = _val('f-dataset');
    const s3Path      = _val('f-s3path').trim();
    const dsName      = _val('f-dsname').trim();

    // What to show in summary for Dataset row
    let datasetDisplay;
    if (datasetType === 'custom') {
      datasetDisplay = dsName
        ? `${dsName} (${s3Path || 'no path set'})`
        : (s3Path || 'custom — no path set');
    } else {
      datasetDisplay = datasetType;
    }

    const items = [
      ['Job ID',   jobId,                                             'color:var(--accent)'],
      ['Task',     _val('f-taskname') || 'Untitled',                  ''],
      ['Dataset',  datasetDisplay,                                     ''],
      ['Priority', document.querySelector('input[name="priority"]:checked')?.value || 'balanced', ''],
      ['Config',   `lr=${_val('f-lr')} h=${_val('f-hidden')} e=${_val('f-epochs')}`, ''],
      ['Compute',  `${_val('f-cloud')} · ${_val('f-instance')}`,      ''],
      ['Budget',   '$' + _val('f-budget'),                            'color:var(--accent2)'],
      ['Fallback', _val('f-fallback'),                                 ''],
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
    const summary = document.getElementById('summary-content');
    const jobId   = summary?.dataset.jobId || ('job-' + Date.now());

    // ── Dataset fields ────────────────────────────────────────────
    // dataset_type: "synthetic-500k" | "synthetic-100k" | "custom"
    // s3_dataset_path: full S3 path, only used when dataset_type=custom
    //   e.g. s3://my-dataset-bucket/train/
    // dataset_name: human-readable label for the dataset
    //
    // These map to job_config.json fields that train.py reads:
    //   cfg["dataset_type"]    → routes to synthetic or S3 loader
    //   cfg["s3_dataset_path"] → parsed for bucket + prefix by _download_s3_dataset()
    //   cfg["dataset_name"]    → display only, shown in dashboard
    const datasetType    = _val('f-dataset') || 'synthetic-500k';
    const s3DatasetPath  = _val('f-s3path').trim();
    const datasetName    = _val('f-dsname').trim() || datasetType;

    // Validate custom dataset before submitting
    if (datasetType === 'custom' && !s3DatasetPath) {
      UI.addLog('log-err', 'Custom dataset selected but no S3 path entered (Step 2)');
      const btn = document.getElementById('btn-next');
      if (btn) { btn.textContent = 'Launch Job'; btn.disabled = false; }
      // Go back to dataset step so user can fill it in
      _goStep(2);
      return;
    }

    if (datasetType === 'custom' && !s3DatasetPath.startsWith('s3://')) {
      UI.addLog('log-err', 'S3 path must start with s3://  e.g. s3://my-bucket/data/');
      _goStep(2);
      return;
    }

    // ── Build request body ────────────────────────────────────────
    const body = {
      // Identity
      job_id:           jobId,
      task_name:        _val('f-taskname') || 'Untitled',

      // Budget / deadline (Step 1)
      max_budget:       parseFloat(_val('f-budget'))  || 2.0,
      deadline_hrs:     parseFloat(_val('f-maxhrs'))  || 8.0,
      priority:         document.querySelector('input[name="priority"]:checked')?.value || 'balanced',

      // Dataset (Step 2)
      // NOTE: "dataset" key kept for backwards compat with old server versions
      dataset:          datasetType,
      dataset_type:     datasetType,        // ← routes synthetic vs S3 in train.py
      s3_dataset_path:  s3DatasetPath,      // ← s3://bucket/prefix/, empty for synthetic
      dataset_name:     datasetName,        // ← display name

      input_dim:        parseInt(_val('f-inputdim'))   || 50,
      num_classes:      parseInt(_val('f-numclasses'))  || 5,

      // Model (Step 3)
      lr:               parseFloat(_val('f-lr'))      || 0.001,
      hidden_dim:       parseInt(_val('f-hidden'))     || 256,
      dropout:          parseFloat(_val('f-dropout'))  || 0.3,
      batch_size:       parseInt(_val('f-batch'))      || 64,
      epochs:           parseInt(_val('f-epochs'))     || 50,
      ckpt_every:       parseInt(_val('f-ckpt'))       || 50,

      // Compute (Step 4)
      cloud:            _val('f-cloud')    || 'gcp',
      instance:         _val('f-instance') || 'e2-standard-4',
      fallback:         _val('f-fallback') || 'migrate',

      // Legacy fields
      model_arch:       'mlp',
      dataset_size:     parseInt(_val('f-inputdim')) || 50,
      min_gpu_mem:      0,
    };

    // ── Show submitting state ─────────────────────────────────────
    const btn = document.getElementById('btn-next');
    if (btn) { btn.textContent = 'Submitting...'; btn.disabled = true; }

    try {
      const resp = await API.submitJob(body);

      if (!resp || resp.error) {
        UI.addLog('log-err', `Launch failed: ${resp?.error || 'no response'}`);
        return;
      }

      // ── Build job card meta ───────────────────────────────────
      const dsLabel = datasetType === 'custom'
        ? `dataset: ${datasetName}`
        : `dataset: ${datasetType}`;

      UI.addJob({
        id:     jobId,
        name:   `${body.task_name} · lr=${body.lr} · h=${body.hidden_dim}`,
        meta:   `GCP ${resp.instance_type || 'e2-standard-4'} · `
              + `$${resp.price_usd_hr?.toFixed(3)}/hr · `
              + `${dsLabel} · ⏳ VM creating (~90s)...`,
        pct:    0,
        status: 'running',
        cloud:  resp.cloud || 'gcp',
      });

      UI.addLog('log-ok',
        `Job ${jobId} queued → GCP ${resp.instance_type} · ${dsLabel}`
      );

      close();

    } catch (e) {
      UI.addLog('log-err', `Submit error: ${e.message}`);
    } finally {
      if (btn) { btn.textContent = 'Launch Job'; btn.disabled = false; }
    }
  }

  return { open, close, next, prev, updateDatasetMeta, _goStep };
})();