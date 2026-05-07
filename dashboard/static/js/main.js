/**
 * static/js/main.js  — updated
 * ──────────────────────────────
 * Added: setInterval(UI.pollActiveJobs, 10_000)
 * This polls /api/jobs/status/<id> every 10s for each active job
 * so progress updates appear without waiting the full 60s refresh.
 */

const INIT_LOGS = [
  ['log-ok',   'Price poller active · 60s interval'],
  ['log-ok',   'BigQuery table: spot_prices.price_history'],
  ['log-ok',   'Checkpointing: GCS + S3 active'],
  ['log-warn', 'Risk model: Phase 2 pending'],
  ['log-info', 'Pareto optimizer: Phase 4 pending'],
];

const AUTO_LOGS = [
  ['log-ok',   'Price poll complete · rows → BigQuery'],
  ['log-info', 'Watcher heartbeat · no termination scheduled'],
  ['log-ok',   'Checkpoint saved · GCS + S3'],
  ['log-warn', 'Price delta detected · AWS g5.xlarge +4.2%'],
];
let _autoIdx = 0;

async function refresh() {
  const { history, summary, latest, preemptionList, statData } = await API.refreshAll();

  const gotRealChart = CHART.applyHistory(history);
  if (summary)        UI.updateStatCards(summary);
  if (summary)        UI.updateNumericCards(summary);
  if (latest)         UI.updatePriceTable(latest);
  if (preemptionList) UI.addPreemptionLogs(preemptionList);
  if (statData)       UI.updateStats(statData);

  const realJobs = await API.jobs();
  if (realJobs && realJobs.length > 0) UI.mergeRealJobs(realJobs);

  if (!gotRealChart && !summary) {
    UI.updateNumericCardsMock(CHART.getCurrentValues());
  }

  CHART.drawPriceChart();
  CHART.drawPareto();
}

function mockTick() {
  if (!API.isLive()) {
    CHART.mockTick();
    UI.updateNumericCardsMock(CHART.getCurrentValues());
    CHART.drawPriceChart();
  }
}

function autoLog() {
  if (!API.isLive()) {
    const e = AUTO_LOGS[_autoIdx % AUTO_LOGS.length];
    UI.addLog(e[0], e[1]);
    _autoIdx++;
  }
}

(function init() {
  UI.startClock();
  UI.renderJobs();
  [...INIT_LOGS].reverse().forEach(([c, m]) => UI.addLog(c, m));

  window.addEventListener('resize', () => {
    CHART.drawPriceChart();
    CHART.drawPareto();
  });

  CHART.drawPriceChart();
  CHART.drawPareto();
  UI.updateNumericCardsMock(CHART.getCurrentValues());

  refresh();
  setInterval(refresh, 60_000);
  setInterval(mockTick, 3_000);
  setInterval(autoLog, 8_000);

  // Poll active jobs every 10s for live progress updates
  setInterval(UI.pollActiveJobs, 10_000);
})();

// Global handlers (called from HTML)
function openModal()         { MODAL.open();             }
function closeModal()        { MODAL.close();            }
function nextStep()          { MODAL.next();             }
function prevStep()          { MODAL.prev();             }
function goStep(n)           { MODAL._goStep(n);         }
function updateDatasetMeta() { MODAL.updateDatasetMeta(); }

document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('modal');
  if (overlay) overlay.addEventListener('click', e => {
    if (e.target === overlay) closeModal();
  });
});