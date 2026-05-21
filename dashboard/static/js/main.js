/**
 * static/js/main.js
 * ──────────────────
 * Entry point — wires API, CHART, UI, and MODAL together.
 *
 * Changes from previous version:
 *   - setInterval(UI.pollActiveJobs, 10_000) added — polls
 *     /api/jobs/status/<id> every 10s for each active job so
 *     progress updates appear without waiting the full 60s refresh.
 *   - refresh() now handles risk data from API.refreshAll()
 *     and passes it to UI.updateRiskBars()
 *   - INIT_LOGS updated to reflect current phase state
 *
 * Load order in index.html:
 *   api.js → chart.js → ui.js → modal.js → main.js
 */

// ── Initial logs ───────────────────────────────────────────────────
const INIT_LOGS = [
  ['log-ok',   'Price poller active · 60s interval'],
  ['log-ok',   'BigQuery table: spot_prices.price_history'],
  ['log-ok',   'Checkpointing: GCS + S3 active'],
  ['log-warn', 'Risk model: Phase 2 pending'],
  ['log-info', 'Pareto optimizer: Phase 4 pending'],
];

// ── Mock auto-log (shown when no API) ─────────────────────────────
const AUTO_LOGS = [
  ['log-ok',   'Price poll complete · rows → BigQuery'],
  ['log-info', 'Watcher heartbeat · no termination scheduled'],
  ['log-ok',   'Checkpoint saved · GCS + S3'],
  ['log-warn', 'Price delta detected · AWS g5.xlarge +4.2%'],
];
let _autoIdx = 0;

// ── Main refresh ───────────────────────────────────────────────────
async function refresh() {
  const { history, summary, latest, preemptionList, statData, risk } =
    await API.refreshAll();

  const gotRealChart = CHART.applyHistory(history);

  if (summary)        UI.updateStatCards(summary);
  if (summary)        UI.updateNumericCards(summary);
  if (latest)         UI.updatePriceTable(latest);
  if (preemptionList) UI.addPreemptionLogs(preemptionList);
  if (statData)       UI.updateStats(statData);
  if (risk)           UI.updateRiskBars(risk);   // ← LSTM + XGBoost risk scores

  // Real jobs from GCS (if available)
  const realJobs = await API.jobs();
   UI.mergeRealJobs(realJobs);

  if (!gotRealChart && !summary) {
    // No real data — keep mock numerics in sync with mock chart
    UI.updateNumericCardsMock(CHART.getCurrentValues());
  }

  CHART.drawPriceChart();
  CHART.drawPareto();
}

// ── Mock tick (keeps chart alive without API) ──────────────────────
function mockTick() {
  if (!API.isLive()) {
    CHART.mockTick();
    UI.updateNumericCardsMock(CHART.getCurrentValues());
    CHART.drawPriceChart();
  }
}

// ── Auto log in mock mode ──────────────────────────────────────────
function autoLog() {
  if (!API.isLive()) {
    const e = AUTO_LOGS[_autoIdx % AUTO_LOGS.length];
    UI.addLog(e[0], e[1]);
    _autoIdx++;
  }
}

// ── Boot ───────────────────────────────────────────────────────────
(function init() {
  UI.startClock();

  // Show loading spinner immediately — don't render empty/demo jobs
  // mergeRealJobs() will replace it once /api/jobs responds
  UI.renderJobs();

  // Seed initial logs (reverse so newest appears at top)
  [...INIT_LOGS].reverse().forEach(([c, m]) => UI.addLog(c, m));

  // Wire up resize
  window.addEventListener('resize', () => {
    CHART.drawPriceChart();
    CHART.drawPareto();
  });

  // First render chart with mock data while API loads
  CHART.drawPriceChart();
  CHART.drawPareto();
  UI.updateNumericCardsMock(CHART.getCurrentValues());

  // Fetch real jobs from GCS immediately on page load —
  // this runs BEFORE the full 60s refresh so jobs appear right away.
  // mergeRealJobs() handles the loading→real transition.
  API.jobs().then(realJobs => {
    if (realJobs) UI.mergeRealJobs(realJobs);
    else {
      // API unreachable — mark loaded so spinner clears
      UI.mergeRealJobs([]);
      UI.addLog('log-warn', 'GCS unreachable — showing empty job list');
    }
  });

  // Full refresh (prices + jobs + risk) every 60s
  refresh();
  setInterval(refresh, 60_000);

  // Mock ticks every 3s so chart moves when no API
  setInterval(mockTick, 3_000);

  // Mock auto-log every 8s
  setInterval(autoLog, 8_000);

  // Poll active jobs every 10s for live progress updates
  setInterval(UI.pollActiveJobs, 10_000);
})();

// ── Global event handlers (called from HTML) ───────────────────────
function openModal()         { MODAL.open();              }
function closeModal()        { MODAL.close();             }
function nextStep()          { MODAL.next();              }
function prevStep()          { MODAL.prev();              }
function goStep(n)           { MODAL._goStep(n);          }
function updateDatasetMeta() { MODAL.updateDatasetMeta(); }

// Close modal when clicking overlay background
document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('modal');
  if (overlay) overlay.addEventListener('click', e => {
    if (e.target === overlay) closeModal();
  });
});