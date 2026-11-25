const drop = $('#drop'), fileIn = $('#file');
fileIn.addEventListener('change', () => $('#fname').textContent = fileIn.files[0] ? `${fileIn.files[0].name} · ${(fileIn.files[0].size / 1024).toFixed(0)} KB` : '');
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) { fileIn.files = ev.dataTransfer.files; fileIn.dispatchEvent(new Event('change')); } });

$('#upload').addEventListener('click', async () => {
  const m = $('#umsg'), btn = $('#upload');
  if (!fileIn.files[0]) return alertBox(m, 'err', 'Choose a CSV file first.');
  if (!/\.csv$/i.test(fileIn.files[0].name)) return alertBox(m, 'err', 'Only .csv files are supported.');
  const fd = new FormData(); fd.append('file', fileIn.files[0]);
  setBusy(btn, true, 'Validating & saving…'); alertBox(m, '', '');
  try {
    const d = await api('/api/upload', {method: 'POST', body: fd}), r = d.report;
    let html = `<strong>Saved to database (upload #${d.upload_id}).</strong><br>${r.rows_stored.toLocaleString()} of ${r.rows_received.toLocaleString()} rows stored · ${r.positives} readmitted &lt;30d · ${r.negatives} others`;
    if (r.rows_duplicate) html += `<br>${r.rows_duplicate} duplicate encounter(s) skipped`;
    if (r.rows_excluded_expired) html += `<br>${r.rows_excluded_expired} in-hospital death encounter(s) excluded (same rule as the original project)`;
    if (r.rows_rejected) html += `<br>${r.rows_rejected} invalid row(s) rejected: ${esc(Object.entries(r.reject_reasons).map(([k, v]) => `${k} (${v})`).join('; '))}`;
    alertBox(m, r.rows_rejected ? 'warn' : 'ok', html);
    $('#count').textContent = `${d.total_records.toLocaleString()} stored record(s) available`; $('#retrain').disabled = false;
    setTimeout(() => location.hash = '', 0); refreshUploads(d);
  } catch (e) { alertBox(m, 'err', esc(e.error || 'Upload failed.')); } finally { setBusy(btn, false); }
});
function refreshUploads() { fetch('/retrain').then(r => r.text()).then(t => { const doc = new DOMParser().parseFromString(t, 'text/html'); $('#uploads').innerHTML = doc.querySelector('#uploads').innerHTML; }); }

function metricRow(label, b, a, key, higherBetter = true) {
  const d = a[key] - b[key], good = higherBetter ? d >= 0 : d <= 0;
  return `<tr><td>${label}</td><td class="num">${f4(b[key])}</td><td class="num">${f4(a[key])}</td><td class="num delta ${Math.abs(d) < 5e-5 ? '' : good ? 'up' : 'down'}">${d >= 0 ? '+' : ''}${d.toFixed(4)}</td></tr>`;
}
function cmpTable(title, c) {
  return `<h3 style="margin-top:14px">${title}</h3><div class="tablewrap"><table><thead><tr><th>Metric</th><th class="num">Before</th><th class="num">After</th><th class="num">Δ</th></tr></thead><tbody>` +
    [['ROC-AUC', 'roc_auc'], ['PR-AUC', 'pr_auc'], ['Recall', 'recall'], ['Precision', 'precision'], ['F1-score', 'f1'], ['Accuracy', 'accuracy']].map(([l, k]) => metricRow(l, c.before, c.after, k)).join('') +
    metricRow('Log-loss', c.before, c.after, 'log_loss', false) + '</tbody></table></div>';
}

$('#retrain').addEventListener('click', async () => {
  const m = $('#rmsg'), btn = $('#retrain'), res = $('#rres');
  setBusy(btn, true, 'Retraining… this can take a few seconds'); alertBox(m, '', ''); res.innerHTML = '<div class="skeleton" style="margin:14px 0"></div><div class="skeleton" style="width:70%"></div>';
  try {
    const d = await api('/api/retrain', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({additional_trees: +$('#trees').value, learning_rate: +$('#lr').value, activate: $('#activate').checked})});
    const mt = d.metrics, ref = mt.reference_test, nv = mt.new_data_validation;
    const drop = ref.after.roc_auc - ref.before.roc_auc;
    alertBox(m, drop < -0.01 ? 'warn' : 'ok',
      `<strong>${esc(d.name)}</strong> trained in ${mt.duration_sec}s on ${d.n_train.toLocaleString()} rows (validated on ${d.n_val.toLocaleString()}); trees ${mt.trees_before} → ${mt.trees_after}. ${d.activated ? 'It is now the active model.' : 'Saved but not activated.'}` +
      (drop < -0.01 ? '<br>⚠ Reference-set ROC-AUC dropped — the upload may differ from the original data. You can roll back below.' : ''));
    res.innerHTML = cmpTable('On held-out part of the new data', nv) + cmpTable('On fixed reference sample (original test split)', ref);
    $('#parent').textContent = d.activated ? d.name : $('#parent').textContent;
    setTimeout(() => location.reload(), 6000);
    res.insertAdjacentHTML('beforeend', '<p class="hint">Refreshing the version table in a few seconds…</p>');
  } catch (e) { res.innerHTML = ''; alertBox(m, 'err', esc(e.error || 'Retraining failed.')); } finally { setBusy(btn, false); }
});

$$('.act').forEach(b => b.addEventListener('click', async () => {
  setBusy(b, true, 'Activating…');
  try { await api(`/api/versions/${b.dataset.id}/activate`, {method: 'POST'}); location.reload(); }
  catch (e) { alert(e.error || 'Could not activate'); setBusy(b, false); }
}));
