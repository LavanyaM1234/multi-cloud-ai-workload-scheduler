/**
 * static/js/modal.js
 * 5-step job submission modal
 * Steps: 1-Job Config  2-Dataset  3-Model & Training  4-Compute  5-Summary
 */

const MODAL = (() => {
  let _step = 1;
  const TOTAL = 5;

  // ── Static dataset metadata ───────────────────────────────────────────────
  const DATASETS = {
    'synthetic-500k': {
      rows: '500,000', features: '50', classes: '5', task: 'classif.', path: '',
    },
    'synthetic-100k': {
      rows: '100,000', features: '50', classes: '5', task: 'classif.', path: '',
    },
    'custom': { rows: '—', features: '—', classes: '—', task: '—', path: '' },
  };

  // ── Min VRAM (GB) derived from param count + precision ───────────────────
  const VRAM_MAP = {
    '<1B':   { fp32: 4,  fp16: 2,  bf16: 2,  int8: 1  },
    '1-7B':  { fp32: 28, fp16: 14, bf16: 14, int8: 7  },
    '7-70B': { fp32: 280,fp16: 140,bf16: 140,int8: 70 },
    '70B+':  { fp32: 999,fp16: 280,bf16: 280,int8: 140},
  };

  function _minGpuMem() {
    const params    = _val('f-params')    || '<1B';
    const precision = _val('f-precision') || 'fp16';
    return (VRAM_MAP[params] || VRAM_MAP['<1B'])[precision] || 0;
  }

  // ── Open / close ─────────────────────────────────────────────────────────
  function open() {
    _step = 1;
    _goStep(1);
    document.getElementById('modal').classList.add('open');

    // Priority buttons
    ['cheapest', 'balanced', 'fastest'].forEach(p => {
      const el = document.getElementById('pri-' + p);
      if (el) el.onclick = () => {
        document.querySelectorAll('.priority-opt').forEach(x => x.classList.remove('selected'));
        el.classList.add('selected');
        el.querySelector('input').checked = true;
      };
    });

    // Training mode toggle
    const modeToggle = document.getElementById('f-train-mode');
    if (modeToggle) {
      modeToggle.onchange = _onTrainModeChange;
      _onTrainModeChange();
    }

    // Spot-only toggle
    const spotToggle = document.getElementById('f-spot-only');
    if (spotToggle) {
      spotToggle.onchange = _onSpotOnlyChange;
      _onSpotOnlyChange();
    }

    // Carbon toggle
    const carbonToggle = document.getElementById('f-carbon');
    if (carbonToggle) {
      carbonToggle.onchange = _onCarbonChange;
      _onCarbonChange();
    }
  }

  function close() {
    document.getElementById('modal').classList.remove('open');
  }

  // ── Step navigation ───────────────────────────────────────────────────────
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

  // ── Toggle helpers ────────────────────────────────────────────────────────

  // Manual vs Auto Sweep mode in Step 3
  function _onTrainModeChange() {
    const mode   = _val('f-train-mode');
    const manual = document.getElementById('manual-fields');
    const sweep  = document.getElementById('sweep-fields');
    if (manual) manual.style.display = (mode === 'manual') ? 'block' : 'none';
    if (sweep)  sweep.style.display  = (mode === 'sweep')  ? 'block' : 'none';
  }

  // Spot-only → show/hide on-demand fallback duration in Step 1
  function _onSpotOnlyChange() {
    const spotOnly    = document.getElementById('f-spot-only')?.value;
    const fallbackWrap = document.getElementById('ondemand-fallback-wrap');
    if (fallbackWrap) fallbackWrap.style.display = (spotOnly === 'no') ? 'block' : 'none';
  }

  // Carbon-aware → show/hide carbon weight selector in Step 4
  function _onCarbonChange() {
    const enabled    = document.getElementById('f-carbon')?.value;
    const weightWrap = document.getElementById('carbon-weight-wrap');
    if (weightWrap) weightWrap.style.display = (enabled === 'yes') ? 'block' : 'none';
  }

  // ── Dataset metadata ──────────────────────────────────────────────────────
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

  // ── Utility ───────────────────────────────────────────────────────────────
  function _val(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
  }

  function _checked(name) {
    return [...document.querySelectorAll(`input[name="${name}"]:checked`)]
      .map(el => el.value);
  }

  // ── Summary (Step 5) ──────────────────────────────────────────────────────
  function _updateSummary() {
    const jobId = _ensureJobId();
    const mode  = _val('f-train-mode') || 'manual';

    const datasetType = _val('f-dataset');
    const s3Path      = _val('f-s3path').trim();
    const dsName      = _val('f-dsname').trim();
    let datasetDisplay;
    if (datasetType === 'custom') {
      datasetDisplay = dsName ? `${dsName} (${s3Path || 'no path'})` : (s3Path || 'custom');
    } else {
      datasetDisplay = datasetType;
    }

    const clouds = _checked('clouds');

    const configStr = mode === 'manual'
      ? `lr=${_val('f-lr')} h=${_val('f-hidden')} e=${_val('f-epochs')} ckpt=${_val('f-ckpt')}`
      : `sweep lr=[${_val('f-sweep-lr-min')}–${_val('f-sweep-lr-max')}] `
      + `trials=${_val('f-sweep-trials')} budget=$${_val('f-sweep-budget')}`;

    const carbonStr = _val('f-carbon') === 'yes'
      ? `on (${_val('f-carbon-weight') || 'balanced'})`
      : 'off';

    const items = [
      ['Job ID',      jobId,                                                      'color:var(--accent)'],
      ['Task',        _val('f-taskname') || 'Untitled',                           ''],
      ['Dataset',     datasetDisplay,                                              ''],
      ['Priority',    document.querySelector('input[name="priority"]:checked')?.value || 'balanced', ''],
      ['Budget',      '$' + _val('f-budget'),                                     'color:var(--accent2)'],
      ['Spot-Only',   _val('f-spot-only') === 'yes' ? 'Yes' : 'No (on-demand fallback ' + _val('f-ondemand-hrs') + 'h)', ''],
      ['Arch',        `${_val('f-arch') || 'MLP'} · ${_val('f-params') || '<1B'} · ${_val('f-paradigm') || 'fine-tuning'}`, ''],
      ['Precision',   _val('f-precision') || 'fp16',                              ''],
      ['Training',    configStr,                                                   ''],
      ['Clouds',      clouds.length ? clouds.join(', ') : 'any',                 ''],
      ['GPU',         _val('f-gpu') === 'yes' ? `Yes (≥${_minGpuMem()}GB VRAM)` : 'No', ''],
      ['Fallback',    _val('f-fallback'),                                         ''],
      ['Carbon-Aware',carbonStr,                                                  ''],
    ];

    const box = document.getElementById('summary-content');
    if (!box) return;
    box.innerHTML = items.map(([label, val, style]) =>
      `<div>
        <span style="color:var(--muted)">${label}</span><br>
        <span style="${style}">${val}</span>
      </div>`
    ).join('');
  }

  // Stable job ID pinned at first summary render
  function _ensureJobId() {
    const box = document.getElementById('summary-content');
    if (box?.dataset.jobId) return box.dataset.jobId;
    const id = 'job-' + new Date().toISOString().slice(5, 10).replace('-', '') +
               '-' + Math.random().toString(36).slice(2, 8);
    if (box) box.dataset.jobId = id;
    return id;
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  async function _submit() {
    const jobId       = _ensureJobId();
    const mode        = _val('f-train-mode') || 'manual';
    const datasetType = _val('f-dataset') || 'synthetic-500k';
    const s3DatasetPath = _val('f-s3path').trim();
    const datasetName   = _val('f-dsname').trim() || datasetType;

    // Validate custom dataset
    if (datasetType === 'custom' && !s3DatasetPath) {
      UI.addLog('log-err', 'Custom dataset selected but no S3 path entered (Step 2)');
      _goStep(2); return;
    }
    if (datasetType === 'custom' && !s3DatasetPath.startsWith('s3://')) {
      UI.addLog('log-err', 'S3 path must start with s3://  e.g. s3://my-bucket/data/');
      _goStep(2); return;
    }

    const preferredClouds = _checked('clouds');

    // Dataset row count for dataset_size (not input_dim)
    const dsRowCount = parseInt(
      (DATASETS[datasetType]?.rows || '50000').replace(/,/g, '')
    ) || 50000;

    // ── Manual-mode training fields ───────────────────────────────
    const manualFields = mode === 'manual' ? {
      lr:         parseFloat(_val('f-lr'))      || 0.001,
      hidden_dim: parseInt(_val('f-hidden'))     || 256,
      dropout:    parseFloat(_val('f-dropout'))  || 0.3,
      batch_size: parseInt(_val('f-batch'))      || 64,
      epochs:     parseInt(_val('f-epochs'))     || 50,
      ckpt_every: parseInt(_val('f-ckpt'))       || 50,
    } : {};

    // ── Sweep-mode training fields ────────────────────────────────
    const sweepFields = mode === 'sweep' ? {
      sweep_lr_min:    parseFloat(_val('f-sweep-lr-min'))   || 0.0001,
      sweep_lr_max:    parseFloat(_val('f-sweep-lr-max'))   || 0.01,
      sweep_hidden:    _checked('sweep-hidden'),             // ['128','256','512']
      sweep_trials:    parseInt(_val('f-sweep-trials'))     || 5,
      sweep_budget:    parseFloat(_val('f-sweep-budget'))   || 5.0,
    } : {};

    const body = {
      // Identity
      job_id:       jobId,
      task_name:    _val('f-taskname') || 'Untitled',
      train_mode:   mode,               // 'manual' | 'sweep'

      // Budget / deadline (Step 1)
      max_budget:       parseFloat(_val('f-budget'))  || 2.0,
      deadline_hrs:     parseFloat(_val('f-maxhrs'))  || 8.0,
      priority:         document.querySelector('input[name="priority"]:checked')?.value || 'balanced',
      spot_only:        _val('f-spot-only') !== 'no',
      ondemand_max_hrs: parseFloat(_val('f-ondemand-hrs')) || 1.0,

      // Dataset (Step 2)
      dataset:          datasetType,
      dataset_type:     datasetType,
      s3_dataset_path:  s3DatasetPath,
      dataset_name:     datasetName,
      dataset_size:     dsRowCount,      // ← fixed: was using input_dim before
      input_dim:        parseInt(_val('f-inputdim'))   || 50,
      num_classes:      parseInt(_val('f-numclasses'))  || 5,

      // Workload identity (Step 3)
      model_arch:        _val('f-arch')      || 'mlp',       // ← no longer hardcoded
      param_count:       _val('f-params')    || '<1B',
      training_paradigm: _val('f-paradigm')  || 'fine-tuning',
      precision:         _val('f-precision') || 'fp16',
      min_gpu_mem:       _minGpuMem(),                       // ← derived, not hardcoded 0

      // Training config (Step 3 — mode-dependent)
      ...manualFields,
      ...sweepFields,

      // Compute (Step 4)
      preferred_clouds:  preferredClouds.length ? preferredClouds : ['aws', 'gcp', 'azure'],
      gpu_required:      _val('f-gpu') === 'yes',
      preferred_regions: _val('f-regions').trim() || '',
      fallback:          _val('f-fallback') || 'migrate',
      carbon_aware:      _val('f-carbon') === 'yes',
      carbon_weight:     _val('f-carbon-weight') || 'balanced',
    };

    const btn = document.getElementById('btn-next');
    if (btn) { btn.textContent = 'Submitting...'; btn.disabled = true; }

    try {
      const resp = await API.submitJob(body);
      if (!resp || resp.error) {
        UI.addLog('log-err', `Launch failed: ${resp?.error || 'no response'}`);
        return;
      }

      const dsLabel = datasetType === 'custom'
        ? `dataset: ${datasetName}`
        : `dataset: ${datasetType}`;

      UI.addJob({
        id:     jobId,
        name:   `${body.task_name} · ${body.model_arch} · ${body.param_count}`,
        meta:   `${resp.cloud || 'gcp'} ${resp.instance_type || 'e2-standard-4'} · `
              + `$${resp.price_usd_hr?.toFixed(3)}/hr · `
              + `${dsLabel} · ⏳ VM creating (~90s)...`,
        pct:    0,
        status: 'running',
        cloud:  resp.cloud || 'gcp',
      });

      UI.addLog('log-ok',
        `Job ${jobId} queued → ${resp.cloud} ${resp.instance_type} · ${dsLabel}`
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