const form = $('#pform'), msg = $('#msg');
const DEFAULTS = {};
$$('#pform [name]').forEach(el => DEFAULTS[el.name] = el.value);
let examples = [], currentActual = null;

function fill(rec) {
  $$('#pform [name]').forEach(el => {
    if (rec[el.name] === undefined) return;
    if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value == rec[el.name]))
      el.insertAdjacentHTML('beforeend', `<option value="${esc(rec[el.name])}">${esc(rec[el.name])} (from dataset)</option>`);
    el.value = rec[el.name];
  });
  $$('.invalid').forEach(e => e.classList.remove('invalid')); $$('.field-err').forEach(e => e.textContent = '');
}
function payload() {
  const d = {}; $$('#pform [name]').forEach(el => d[el.name] = el.value);
  // medications not shown in the form keep the loaded example's value (default 'No')
  Object.entries(currentExtra).forEach(([k, v]) => { if (!(k in d)) d[k] = v; });
  return d;
}
let currentExtra = {};

api('/api/examples').then(list => {
  examples = list;
  const sel = $('#example');
  list.forEach((e, i) => sel.insertAdjacentHTML('beforeend',
    `<option value="${i}">Encounter ${e.encounter_id} — age ${esc(e.record.age)}, ${e.record.number_inpatient} prior inpatient stays</option>`));
}).catch(() => {});
$('#example').addEventListener('change', ev => {
  const i = ev.target.value;
  if (i === '') { currentActual = null; currentExtra = {}; return; }
  const e = examples[i]; currentActual = e.actual; currentExtra = e.record; fill(e.record);
  alertBox(msg, 'info', `Loaded real encounter <strong>${e.encounter_id}</strong>. Press “Predict” to compare the model output with the recorded outcome.`);
});
$('#reset').addEventListener('click', () => { fill(DEFAULTS); currentActual = null; currentExtra = {}; $('#example').value = ''; alertBox(msg, '', ''); $('#result').classList.remove('show'); $('#placeholder').style.display = ''; });

form.addEventListener('submit', async ev => {
  ev.preventDefault();
  $$('.invalid').forEach(e => e.classList.remove('invalid')); $$('.field-err').forEach(e => e.textContent = '');
  alertBox(msg, '', '');
  const btn = $('#go'); setBusy(btn, true, 'Predicting…');
  try {
    const {result: r} = await api('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload())});
    show(r);
  } catch (e) {
    if (e.fields) Object.entries(e.fields).forEach(([k, v]) => { const el = form.elements[k]; if (el) { el.classList.add('invalid'); const fe = el.parentElement.querySelector('.field-err'); if (fe) fe.textContent = v; } });
    alertBox(msg, 'err', esc(e.error || 'Something went wrong.'));
  } finally { setBusy(btn, false); }
});

function show(r) {
  $('#placeholder').style.display = 'none'; $('#result').classList.add('show');
  $('#prob').textContent = pct(r.probability);
  const bar = $('#bar'); bar.style.width = (r.probability * 100) + '%';
  bar.style.background = r.label ? 'var(--bad)' : (r.probability >= .35 ? 'var(--warn)' : 'var(--good)');
  const k = $('#klass'); k.textContent = r.label ? 'Predicted: readmitted < 30 days' : 'Predicted: not readmitted < 30 days';
  k.className = 'badge ' + (r.label ? 'hi' : 'lo'); $('#band').textContent = r.risk_band + ' risk';
  $('#modelname').textContent = 'Model: ' + r.model_name;
  const a = $('#actual');
  if (currentActual) {
    const pos = currentActual === '<30', ok = pos === !!r.label;
    a.style.display = ''; a.innerHTML = `Recorded outcome in dataset: <strong>${esc(currentActual === 'NO' ? 'not readmitted' : currentActual === '>30' ? 'readmitted after 30 days' : 'readmitted within 30 days')}</strong> — model ${ok ? '<span class="delta up">agrees ✓</span>' : '<span class="delta down">disagrees ✗</span>'}`;
  } else a.style.display = 'none';
  const mx = Math.max(...r.factors.map(f => Math.abs(f.effect)), .01);
  $('#factors').innerHTML = r.factors.map(f => {
    const w = Math.abs(f.effect) / mx * 50, up = f.effect > 0;
    return `<div class="factor"><div class="name">${esc(f.feature)}</div><div class="bar"><i style="${up ? 'left:50%' : 'right:50%'};width:${w}%;background:var(${up ? '--bad' : '--good'})"></i></div><div class="small muted" style="width:70px">${up ? '▲ raises' : '▼ lowers'}</div></div>`;
  }).join('') || '<span class="muted">No dominant factors.</span>';
  if (innerWidth < 900) $('#resultcard').scrollIntoView({behavior: 'smooth'});
}

// ---- batch prediction -----------------------------------------------------------------
let lastRows = [];
$('#bgo').addEventListener('click', async () => {
  const f = $('#bfile').files[0], m = $('#bmsg'), btn = $('#bgo');
  if (!f) return alertBox(m, 'err', 'Choose a CSV file first.');
  const fd = new FormData(); fd.append('file', f); setBusy(btn, true, 'Predicting…'); alertBox(m, '', ''); $('#bres').innerHTML = '';
  try {
    const d = await api('/api/predict-batch', {method: 'POST', body: fd}); lastRows = d.rows;
    alertBox(m, 'ok', `${d.summary.rows} rows scored · ${d.summary.flagged_high_risk} flagged high risk · mean probability ${pct(d.summary.mean_probability)}`);
    $('#bres').innerHTML = `<div class="tablewrap" style="max-height:260px;overflow:auto"><table><thead><tr><th>#</th><th>Encounter</th><th class="num">Prob.</th><th>Class</th></tr></thead><tbody>` +
      d.rows.slice(0, 100).map(r => `<tr><td>${r.row}</td><td>${esc(r.encounter_id)}</td><td class="num">${pct(r.probability)}</td><td><span class="badge ${r.label ? 'hi' : 'lo'}">${r.label ? 'High' : 'Lower'}</span></td></tr>`).join('') + '</tbody></table></div>' + (d.rows.length > 100 ? '<p class="hint">Showing first 100 rows; download for all.</p>' : '');
    $('#bdl').style.display = '';
  } catch (e) { alertBox(m, 'err', esc(e.error || 'Batch prediction failed.')); } finally { setBusy(btn, false); }
});
$('#bdl').addEventListener('click', () => {
  const csv = 'row,encounter_id,probability,predicted_readmit_30d,risk_band\n' + lastRows.map(r => [r.row, r.encounter_id, r.probability, r.label, r.risk_band].join(',')).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})); a.download = 'predictions.csv'; a.click();
});
