// ADO display mode and status clock, shared by every page.
// Loaded as a blocking script in <head> so the palette is chosen before first paint.
// Renderers that draw their own colours listen for the `ado:display` event.
(() => {
  const KEY = 'ado.display';
  const MODES = ['night', 'day'];
  const root = document.documentElement;

  function stored() {
    try {
      const value = localStorage.getItem(KEY);
      return MODES.includes(value) ? value : null;
    } catch {
      return null;
    }
  }

  function apply(mode) {
    const changed = root.dataset.display !== mode;
    root.dataset.display = mode;
    const toggle = document.getElementById('display-toggle');
    if (toggle) {
      const next = mode === 'night' ? 'day' : 'night';
      toggle.setAttribute('aria-label', `Display mode: ${mode}. Switch to ${next}.`);
      toggle.title = `Switch to the ${next} display (D)`;
    }
    const theme = document.querySelector('meta[name="theme-color"]');
    if (theme) theme.content = mode === 'day' ? '#eceef2' : '#090e16';
    if (changed) dispatchEvent(new CustomEvent('ado:display', {detail: {mode}}));
  }

  function toggle() {
    const mode = root.dataset.display === 'day' ? 'night' : 'day';
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      // The choice still applies to this page.
    }
    apply(mode);
  }

  apply(stored() ?? 'night');

  function startClock(element) {
    const format = new Intl.DateTimeFormat(undefined, {hour: 'numeric', minute: '2-digit'});
    const tick = () => {
      const now = new Date();
      const label = format.format(now);
      if (element.textContent !== label) element.textContent = label;
      element.dateTime = now.toISOString().slice(0, 16);
      setTimeout(tick, 60000 - (now.getSeconds() * 1000 + now.getMilliseconds()) + 50);
    };
    tick();
  }

  function ready() {
    apply(root.dataset.display);
    document.getElementById('display-toggle')?.addEventListener('click', toggle);
    const clock = document.getElementById('display-clock');
    if (clock) startClock(clock);
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.key.toLowerCase() !== 'd') return;
      const target = event.target;
      if (/^(INPUT|SELECT|TEXTAREA|VIDEO)$/.test(target?.tagName) || target?.isContentEditable) return;
      toggle();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
})();
