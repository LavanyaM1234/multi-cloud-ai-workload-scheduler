/**
 * static/js/api.js
 *

 */

const API = (() => {
  const BASE            = 'http://localhost:5050';
  const TIMEOUT         = 12000;
  const RISK_TIMEOUT    = 300_000;
  const JOBS_TIMEOUT = 300_000;
  const SUBMIT_TIMEOUT  = 30000;
  const HISTORY_TIMEOUT = 30_000;

  let _useMock     = true;
  let _wasEverLive = false;
  let _lastOk      = null;

  async function _get(path, timeout = TIMEOUT) {
    try {
      const res = await fetch(BASE + path, {
        signal: AbortSignal.timeout(timeout),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data   = await res.json();
      _lastOk      = new Date();
      _useMock     = false;
      _wasEverLive = true;
      return data;
    } catch (e) {
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

  function _setStatus(state, msg) {
    const el = document.getElementById('api-status-badge');
    if (!el) return;
    if (state === 'live')  { el.textContent = 'LIVE DATA';    el.className = 'badge badge-live'; }
    if (state === 'stale') { el.textContent = 'RECONNECTING'; el.className = 'badge badge-warn'; }
    if (state === 'mock')  { el.textContent = 'MOCK DATA';    el.className = 'badge badge-mock'; }
    el.title = msg;
  }

  const priceHistory = () => _get('/api/prices/history', HISTORY_TIMEOUT);
  const priceSummary = () => _get('/api/prices/summary');
  const latestPrices = () => _get('/api/prices/latest');
  const preemptions  = () => _get('/api/prices/preemptions');
  const jobs         = () => _get('/api/jobs', JOBS_TIMEOUT);
  const stats        = () => _get('/api/stats');
  const health       = () => _get('/api/health');
  const submitJob    = body => _post('/api/jobs/submit', body);
  const riskScores   = () => _get('/api/risk', RISK_TIMEOUT);
  const fetchRisk    = () => _get('/api/risk', RISK_TIMEOUT);
  const fetchPareto = () => _get('/api/pareto', 300_000);  // 5 min, same as cache TTL

  async function refreshAll() {
    const [history, summary, latest, preemptionList, statData, realJobs] =
      await Promise.all([
        priceHistory(),
        priceSummary(),
        latestPrices(),
        preemptions(),
        stats(),
        jobs(),
      ]);

    const results   = [history, summary, latest, preemptionList, statData, realJobs];
    const anyOk     = results.some(r => r !== null);
    const allFailed = results.every(r => r === null);

    if (anyOk) {
      _useMock     = false;
      _wasEverLive = true;
      _lastOk      = new Date();
      _setStatus('live', 'Connected — live data');
    } else if (allFailed && _wasEverLive) {
      _setStatus('stale', 'Reconnecting…');
    } else if (allFailed) {
      _useMock = true;
      _setStatus('mock', 'API unreachable — using mock data');
    }

    return { history, summary, latest, preemptionList, statData, realJobs };
  }

  return {
    priceHistory, priceSummary, latestPrices, preemptions,
    jobs, stats, health, submitJob, riskScores, fetchRisk, fetchPareto, refreshAll,
    isLive:      () => !_useMock,
    wasEverLive: () => _wasEverLive,
    lastOk:      () => _lastOk,
  };
})();