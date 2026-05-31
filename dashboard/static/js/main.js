/**
 * static/js/main.js
 *
 * Changes vs previous version:
 *   [1] risk removed from refreshAll() — it runs via refreshRisk() independently.
 *       This fixes the 4-minute page load since risk (LSTM+XGBoost × N) was
 *       blocking the entire Promise.all.
 *   [2] visibilitychange handler added — triggers an immediate refresh when the
 *       user switches back to the tab, so the chart/badge never stays stale.
 *   [3] All updateNumericCards() / updateNumericCardsMock() calls replaced
 *       with UI.updateHealthStrip(CHART.getCurrentValues()) — retained from prior fix.
 *   [4] CHART.drawSparklines() is now called inside CHART.drawPriceChart() — retained.
 *   [5] pollActiveJobs race fix retained: first tick delayed 15s after boot.
 */

const INIT_LOGS = [
  ['log-ok',   'Price poller active · 60s interval'],
  ['log-ok',   'BigQuery table: spot_prices.price_history'],
  ['log-ok',   'Checkpointing: S3 active'],
];

const AUTO_LOGS = [
  ['log-ok',   'Price poll complete · rows → BigQuery'],
  ['log-info', 'Watcher heartbeat · no termination scheduled'],
  ['log-ok',   'Checkpoint saved · S3'],
  ['log-warn', 'Price delta detected · AWS g5.xlarge +4.2%'],
];
let _autoIdx = 0;

let _lastApiMeta = null;
const HISTORY_TIMEOUT = 30_000;  // _fetch_live_prices hits AWS EC2 API — needs time


async function refreshRisk() {
  const panel = document.getElementById('risk-panel-body');
  const badge = panel?.closest('.panel')?.querySelector('.badge');

  if (badge && badge.textContent !== 'loading…') {
    badge.textContent = 'refreshing…';
    badge.className   = 'badge badge-warn';
  }

  const risk = await (API.fetchRisk ? API.fetchRisk() : API.riskScores());
  console.log('[risk] refreshed:', risk?.length ?? risk?.error ?? 'null');

  if (risk && !risk.error && Array.isArray(risk) && risk.length > 0) {
    UI.updateRiskBars(risk);
    UI.addLog('log-ok', `Risk scores updated · ${risk.length} instances scored`);
  } else if (risk && Array.isArray(risk) && risk.length === 0) {
    UI.addLog('log-warn', 'Risk: no instances scored');
  } else {
    if (badge) { badge.textContent = 'failed'; badge.className = 'badge badge-warn'; }
    UI.addLog('log-warn', 'Risk refresh failed — will retry in 120s');
  }
}

async function refreshPareto() {
  const data = await API.fetchPareto();
  if (data && !data.error) {
    CHART.drawPareto(data);
    UI.addLog('log-ok', `Pareto updated · ${data.pareto_set?.length ?? 0} non-dominated`);
  }
}

async function refresh() {
  // Risk is intentionally absent here — handled by refreshRisk() every 120s
  const { history, summary, latest, preemptionList, statData, realJobs } =
    await API.refreshAll();

  const gotRealChart = CHART.applyHistory(history);
console.log('[history] applied:', gotRealChart, 'timestamps:', history?.timestamps?.length, 'aws sample:', history?.aws?.slice(-3));

  if (history?.meta && history?.stats) {
    _lastApiMeta = {};
    for (const cloud of ['aws', 'gcp', 'azure']) {
      _lastApiMeta[cloud] = {
        ...history.meta[cloud],
        ...history.stats[cloud],
      };
    }
  }

  if (summary)        UI.updateStatCards(summary);
  if (latest)         UI.updatePriceTable(latest);
  if (preemptionList) UI.addPreemptionLogs(preemptionList);
  if (statData)       UI.updateStats(statData);

  UI.mergeRealJobs(realJobs || []);
  if (!realJobs && API.isLive()) {
    UI.addLog('log-warn', 'Jobs fetch failed — will retry in 60s');
  }

  UI.updateHealthStrip(CHART.getCurrentValues(), _lastApiMeta, summary);
  CHART.drawPriceChart();
console.log('[chart] canvas size:', document.getElementById('price-chart')?.offsetWidth);
  CHART.drawPareto();
}

function mockTick() {
  if (!API.isLive() && !API.wasEverLive()) {
    CHART.mockTick();
    UI.updateHealthStrip(CHART.getCurrentValues(), _lastApiMeta);
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

  ;[...INIT_LOGS].reverse().forEach(([c, m]) => UI.addLog(c, m));

  window.addEventListener('resize', () => {
    CHART.drawPriceChart();
    CHART.drawPareto();
    
  });

  // Refresh immediately when user switches back to this tab
  // — fixes the stale chart / RECONNECTING badge staying up
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      refresh();
    }
  });

  CHART.drawPriceChart();
  CHART.drawPareto();
  // Pareto — first call after 3s so risk/price load first
  setTimeout(() => {
    refreshPareto();
    setInterval(refreshPareto, 300_000);  // refresh every 5 min (matches cache TTL)
  }, 3000);
  UI.updateHealthStrip(CHART.getCurrentValues());

  refresh();
  setInterval(refresh, 60_000);

  // Risk runs independently — first call after 2s so page renders first
  setTimeout(() => {
    refreshRisk();
    setInterval(refreshRisk, 120_000);
  }, 2000);

  setInterval(() => { if (!API.isLive()) CHART.mockTick(); }, 3_000);
  setInterval(autoLog, 8_000);

  setTimeout(() => {
    UI.pollActiveJobs();
    setInterval(UI.pollActiveJobs, 10_000);
  }, 15_000);
})();

function openModal()         { MODAL.open();              }
function closeModal()        { MODAL.close();             }
function nextStep()          { MODAL.next();              }
function prevStep()          { MODAL.prev();              }
function goStep(n)           { MODAL._goStep(n);          }
function updateDatasetMeta() { MODAL.updateDatasetMeta(); }

document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('modal');
  if (overlay) overlay.addEventListener('click', e => {
    if (e.target === overlay) closeModal();
  });
});