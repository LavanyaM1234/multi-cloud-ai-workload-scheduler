/**
 * static/js/main.js
 * ──────────────────
 * Entry point — wires API, CHART, UI, and MODAL together.
 *
 * Runs on page load. Controls the refresh cycle.
 *
 * Load order in index.html:
 *   api.js → chart.js → ui.js → modal.js → main.js
 */

// ── Initial mock logs ──────────────────────────────────────────────
const INIT_LOGS = [
  ['log-ok',   'Price poller active · 60s interval'],
  ['log-ok',   'BigQuery table: spot_prices.price_history'],
  ['log-warn', 'Risk model: Phase 2 pending'],
  ['log-info', 'Pareto optimizer: Phase 4 pending'],
  ['log-ok',   'Checkpointing: GCS + S3 active'],
];

// ── Mock auto-log (shown when no API) ─────────────────────────────
const AUTO_LOGS = [
  ['log-ok',   'Price poll complete · rows → BigQuery'],
  ['log-info', 'Watcher heartbeat · no termination scheduled'],
  ['log-ok',   'Checkpoint saved · demo-job-001'],
  ['log-warn', 'Price delta detected · AWS g5.xlarge +4.2%'],
];
let _autoIdx = 0;

// ── Main refresh ───────────────────────────────────────────────────
async function refresh() {
  const { history, summary, latest, preemptionList, statData } = await API.refreshAll();

  const gotRealChart = CHART.applyHistory(history);

  if (summary)         UI.updateStatCards(summary);
  if (summary)         UI.updateNumericCards(summary);
  if (latest)          UI.updatePriceTable(latest);
  if (preemptionList)  UI.addPreemptionLogs(preemptionList);
  if (statData)        UI.updateStats(statData);

  // Real jobs from GCS (if available)
  const realJobs = await API.jobs();
  if (realJobs && realJobs.length > 0) UI.mergeRealJobs(realJobs);

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
  UI.renderJobs();

  // Seed initial logs
  [...INIT_LOGS].reverse().forEach(([c, m]) => UI.addLog(c, m));

  // Wire up resize
  window.addEventListener('resize', () => {
    CHART.drawPriceChart();
    CHART.drawPareto();
  });

  // First render with mock data
  CHART.drawPriceChart();
  CHART.drawPareto();
  UI.updateNumericCardsMock(CHART.getCurrentValues());

  // Try real API immediately, then every 60s
  refresh();
  setInterval(refresh, 60_000);

  // Mock ticks every 3s so chart moves
  setInterval(mockTick, 3_000);

  // Mock auto-log every 8s
  setInterval(autoLog, 8_000);
})();

// ── Global event handlers (called from HTML) ───────────────────────
function openModal()   { MODAL.open();  }
function closeModal()  { MODAL.close(); }
function nextStep()    { MODAL.next();  }
function prevStep()    { MODAL.prev();  }
function goStep(n)     { MODAL._goStep?.(n); }  // expose for tab clicks
function updateDatasetMeta() { MODAL.updateDatasetMeta(); }

// Close modal when clicking overlay background
document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('modal');
  if (overlay) overlay.addEventListener('click', e => {
    if (e.target === overlay) closeModal();
  });
});
