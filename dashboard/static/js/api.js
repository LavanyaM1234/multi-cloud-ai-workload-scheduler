/**
 * static/js/api.js
 * ─────────────────
 * All network calls to the Flask backend (api/server.py).
 *
 * Change API_BASE to your Flask server's address:
 *   - Local:  "http://localhost:5050"
 *   - VM:     "http://<ORCHESTRATOR_VM_IP>:5050"
 *
 * If the server is unreachable, every function returns null
 * and the dashboard falls back to mock data automatically.
 */

const API = (() => {
  // ── Config ────────────────────────────────────────────────────────
  // Change this to your Flask server URL
  const BASE    = 'http://localhost:5050';
  const TIMEOUT = 6000; // ms

  let _useMock  = true;  // flips to false when first real response arrives
  let _lastOk   = null;  // timestamp of last successful API call

  // ── Internal fetch helper ─────────────────────────────────────────
  async function _get(path) {
    try {
      const res = await fetch(BASE + path, {
        signal: AbortSignal.timeout(TIMEOUT),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      _lastOk    = new Date();
      _useMock   = false;
      _setStatus(true, 'Connected — live BigQuery data');
      return data;
    } catch (e) {
      if (!_useMock) {
        // Was working, now failing — warn user
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
        signal:  AbortSignal.timeout(TIMEOUT),
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
  const priceHistory   = () => _get('/api/prices/history');

  /** min/max/avg/current per cloud over last 3 hours */
  const priceSummary   = () => _get('/api/prices/summary');

  /** latest price row per instance type */
  const latestPrices   = () => _get('/api/prices/latest');

  /** recent preempted=TRUE rows */
  const preemptions    = () => _get('/api/prices/preemptions');

  /** active jobs from GCS job_state.json files */
  const jobs           = () => _get('/api/jobs');

  /** high-level stats (total rows, preemption count) */
  const stats          = () => _get('/api/stats');

  /** health check — BigQuery reachable? */
  const health         = () => _get('/api/health');

  /** submit job to scheduler */
  const submitJob      = body => _post('/api/jobs/submit', body);

  /** fetch everything at once — called every 60s */
  async function refreshAll() {
    const [history, summary, latest, preemptionList, statData] = await Promise.all([
      priceHistory(),
      priceSummary(),
      latestPrices(),
      preemptions(),
      stats(),
    ]);
    return { history, summary, latest, preemptionList, statData };
  }

  return {
    priceHistory, priceSummary, latestPrices, preemptions,
    jobs, stats, health, submitJob, refreshAll,
    isLive: () => !_useMock,
    lastOk: () => _lastOk,
  };
})();
