/**
 * static/js/api.js
 * ─────────────────
 * All network calls to the Flask backend (api/server.py).
 *
 * Change API_BASE to your Flask server's address:
 *   - Local:  "http://localhost:5050"
 *   - VM:     "http://<ORCHESTRATOR_VM_IP>:5050"
 *
 * Changes from previous version:
 *   - riskScores() added — calls /api/risk (LSTM + XGBoost scores)
 *   - refreshAll() includes riskScores() in Promise.all
 *   - _post() timeout increased to 30s for job submit
 *     (VM creation is non-blocking now but keeps headroom)
 */

const API = (() => {
  // ── Config ────────────────────────────────────────────────────────
  const BASE           = 'http://localhost:5050';
  const TIMEOUT        = 6000;   // ms — for GET requests
  const SUBMIT_TIMEOUT = 30000;  // ms — for POST /api/jobs/submit

  let _useMock = true;   // flips to false when first real response arrives
  let _lastOk  = null;   // timestamp of last successful API call

  // ── Internal fetch helpers ────────────────────────────────────────

  async function _get(path) {
    try {
      const res = await fetch(BASE + path, {
        signal: AbortSignal.timeout(TIMEOUT),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      _lastOk  = new Date();
      _useMock = false;
      _setStatus(true, 'Connected — live BigQuery data');
      return data;
    } catch (e) {
      if (!_useMock) {
        _setStatus(false, `API unreachable: ${e.message}`);
      }
      return null;
    }
  }

  async function _post(path, body) {
    try {
      const res = await fetch(BASE + path, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal:  AbortSignal.timeout(SUBMIT_TIMEOUT),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { error: e.message };
    }
  }

  // ── Status badge ──────────────────────────────────────────────────

  function _setStatus(ok, msg) {
    const el = document.getElementById('api-status-badge');
    if (!el) return;
    el.textContent = ok ? 'LIVE DATA' : 'MOCK DATA';
    el.className   = 'badge ' + (ok ? 'badge-live' : 'badge-mock');
    el.title       = msg;
  }

  // ── Endpoints ─────────────────────────────────────────────────────

  /** 30-point price time series per cloud */
  const priceHistory = () => _get('/api/prices/history');

  /** min/max/avg/current per cloud over last 3 hours */
  const priceSummary = () => _get('/api/prices/summary');

  /** latest price row per instance type */
  const latestPrices = () => _get('/api/prices/latest');

  /** recent preempted=TRUE rows from BigQuery */
  const preemptions  = () => _get('/api/prices/preemptions');

  /** active jobs from GCS job_state.json files */
  const jobs         = () => _get('/api/jobs');

  /** high-level stats (total rows, preemption count) */
  const stats        = () => _get('/api/stats');

  /** health check — BigQuery reachable? */
  const health       = () => _get('/api/health');

  /** submit job to scheduler — returns immediately, VM created in background */
  const submitJob    = body => _post('/api/jobs/submit', body);

  /**
   * Preemption risk scores from LSTM + XGBoost model.
   * Returns list of {cloud, instance_type, region, az, risk: 0.0–1.0}
   * Falls back to null if /api/risk not yet available (model not loaded).
   */
  const riskScores   = () => _get('/api/risk');

  /**
   * Fetch everything at once — called every 60s by main.js refresh().
   * risk is included so the risk bars update on every full refresh.
   */
  async function refreshAll() {
    const [history, summary, latest, preemptionList, statData, risk] =
      await Promise.all([
        priceHistory(),
        priceSummary(),
        latestPrices(),
        preemptions(),
        stats(),
        riskScores(),   // ← LSTM + XGBoost preemption risk scores
      ]);
    return { history, summary, latest, preemptionList, statData, risk };
  }

  return {
    priceHistory, priceSummary, latestPrices, preemptions,
    jobs, stats, health, submitJob, riskScores, refreshAll,
    isLive: () => !_useMock,
    lastOk: () => _lastOk,
  };
})();