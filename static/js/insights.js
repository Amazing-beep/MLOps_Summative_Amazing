api('/api/insights').then(d => {
  const ds = d.dataset, ov = ds.prevalence_pct;
  $('#kpis').innerHTML = [['Raw encounters', ds.raw_rows.toLocaleString(), `${ds.raw_columns} columns`], ['Unique patients', ds.unique_patients.toLocaleString(), 'repeat visits are grouped in splits'],
    ['Eligible encounters', ds.eligible_rows.toLocaleString(), 'after excluding in-hospital deaths'], ['30-day readmission rate', ov + '%', `${ds.positives.toLocaleString()} positive encounters`]]
    .map(([l, v, s]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${s}</div></div>`).join('');
  const rows = a => a.map(r => ({label: r.label, value: r.rate, note: `${r.n.toLocaleString()} encounters`}));
  const opt = {max: 25, ref: ov};
  hbars($('#c_inp'), rows(d.by_prior_inpatient).map(r => ({...r, label: r.label + ' prior stays'})), opt);
  hbars($('#c_diag'), rows(d.by_diag1), opt); hbars($('#c_age'), rows(d.by_age), opt);
  hbars($('#c_los'), rows(d.by_los), opt); hbars($('#c_meds'), rows(d.by_meds), opt);
  hbars($('#c_a1c'), rows(d.by_a1c), opt); hbars($('#c_ins'), rows(d.by_insulin), opt);
  const neg = ds.eligible_rows - ds.positives;
  hbars($('#c_cls'), [{label: 'Not readmitted <30d', value: neg / ds.eligible_rows * 100}, {label: 'Readmitted <30d', value: ds.positives / ds.eligible_rows * 100}], {max: 100});
  hbars($('#c_imp'), d.feature_importance.map(f => ({label: f.feature, value: f.gain})), {fmt: v => v.toFixed(1) + '%'});
}).catch(e => $('#kpis').innerHTML = `<div class="alert err">Could not load insights: ${esc(e.error || 'error')}</div>`);
