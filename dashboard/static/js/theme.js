/**
 * static/js/theme.js
 * ───────────────────
 * Dark / Light theme toggle.
 * Persists choice in localStorage so it survives page refreshes.
 *
 * Usage: include this BEFORE main.js in index.html
 *   <script src="/static/js/theme.js"></script>
 *
 * The toggle button HTML (already in index.html header):
 *   <button class="theme-toggle" onclick="THEME.toggle()" id="theme-btn">
 *     <div class="toggle-track"><div class="toggle-thumb"></div></div>
 *     <span class="toggle-label" id="theme-label">Light</span>
 *   </button>
 */

const THEME = (() => {

  const STORAGE_KEY = 'scheduler-theme';

  function _apply(mode) {
    // mode = 'dark' | 'light'
    if (mode === 'light') {
      document.body.classList.add('light');
    } else {
      document.body.classList.remove('light');
    }
    // Update toggle label
    const label = document.getElementById('theme-label');
    if (label) label.textContent = mode === 'light' ? 'Dark' : 'Light';
    // Redraw canvas charts — they read CSS vars for colors
    if (typeof CHART !== 'undefined') {
      CHART.drawPriceChart();
      CHART.drawPareto();
    }
  }

  function toggle() {
    const isLight = document.body.classList.contains('light');
    const next    = isLight ? 'dark' : 'light';
    localStorage.setItem(STORAGE_KEY, next);
    _apply(next);
  }

  function init() {
    // Read saved preference, default to dark
    const saved = localStorage.getItem(STORAGE_KEY) || 'dark';
    _apply(saved);
  }

  return { toggle, init };
})();

// Run immediately so there's no flash of wrong theme
THEME.init();