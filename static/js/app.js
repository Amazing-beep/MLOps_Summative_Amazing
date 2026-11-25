// Shared helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
const f4 = x => (x === null || x === undefined) ? '–' : Number(x).toFixed(4);

async function api(url, opts = {}) {
  let res;
  try { res = await fetch(url, opts); }
  catch (e) { throw {error: 'Cannot reach the server. Check that the app is running.'}; }
  let data = {};
  try { data = await res.json(); } catch (e) { data = {error: `Unexpected server response (${res.status}).`}; }
  if (!res.ok || data.ok === false) throw data;
  return data;
}
function setBusy(btn, busy, label) {
  if (busy) { btn.dataset.label = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> ${label || 'Working…'}`; }
  else { btn.disabled = false; btn.innerHTML = btn.dataset.label || btn.innerHTML; }
}
function alertBox(el, kind, html) { el.innerHTML = html ? `<div class="alert ${kind}" role="alert">${html}</div>` : ''; }

// Horizontal bar chart (dependency-free). rows: [{label, value, note}]
function hbars(el, rows, {max, fmt = v => v.toFixed(1) + '%', ref} = {}) {
  const m = max || Math.max(...rows.map(r => r.value)) * 1.1 || 1;
  el.innerHTML = rows.map(r => `<div class="hbar" title="${esc(r.note || '')}"><div class="lab">${esc(r.label)}</div>
    <div class="track"><div class="fill" style="width:${Math.max(r.value / m * 100, 0.5)}%"></div></div><div class="val">${fmt(r.value)}</div></div>`).join('') +
    (ref !== undefined ? `<div class="refline">Overall readmission rate: ${ref}%</div>` : '');
}
