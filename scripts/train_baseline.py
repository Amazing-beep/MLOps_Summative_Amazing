"""Rebuild the ORIGINAL pre-trained model and the app's static assets.

Re-runs the exact pipeline from the Intro-to-ML notebook (EXP-ML-05, cost-
sensitive LightGBM, patient-grouped 70/15/15 split, seed 42) and writes:

  models/best_traditional_model.joblib   fitted ColumnTransformer + LightGBM
  models/base_metrics.json               validation + held-out test metrics
  data/insights.json                     dataset aggregates for the Insights page
  data/examples.json                     real encounters for the prediction demo
  data/reference_test_sample.csv         labelled held-out sample (before/after retrain)
  data/sample_retrain_*.csv              labelled files to demo uploads (validation split)
  data/sample_unlabeled.csv              unlabeled file for batch prediction

Usage: python scripts/train_baseline.py [path/to/diabetic_data.csv]
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from readmission import config  # noqa: E402
from readmission.data import apply_target_and_exclusions, load_raw, patient_grouped_split  # noqa: E402
from readmission.features import build_preprocessor, engineer_features, parse_icd9_code, RAW_COLUMNS  # noqa: E402
from readmission.metrics import compute_metrics  # noqa: E402


def rate_table(df, col, order=None):
    g = df.groupby(col)['target'].agg(['mean', 'count'])
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    return [{'label': str(i), 'rate': round(float(r['mean']) * 100, 2), 'n': int(r['count'])}
            for i, r in g.iterrows()]


def build_insights(df_raw, df_clean, importances):
    df = df_clean.copy()
    df['diag_1_group'] = df['diag_1'].apply(parse_icd9_code)
    df['inpatient_bucket'] = pd.cut(df['number_inpatient'], [-1, 0, 1, 2, 100],
                                    labels=['0', '1', '2', '3+']).astype(str)
    df['los_bucket'] = pd.cut(df['time_in_hospital'], [0, 2, 4, 7, 100],
                              labels=['1-2 d', '3-4 d', '5-7 d', '8+ d']).astype(str)
    df['meds_bucket'] = pd.cut(df['num_medications'], [-1, 10, 15, 20, 200],
                               labels=['0-10', '11-15', '16-20', '21+']).astype(str)
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                 '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    return {
        'dataset': {
            'raw_rows': int(len(df_raw)), 'raw_columns': int(df_raw.shape[1]),
            'unique_patients': int(df_raw['patient_nbr'].nunique()),
            'eligible_rows': int(len(df_clean)),
            'eligible_patients': int(df_clean['patient_nbr'].nunique()),
            'positives': int(df_clean['target'].sum()),
            'prevalence_pct': round(float(df_clean['target'].mean()) * 100, 2),
            'raw_label_counts': {k: int(v) for k, v in df_raw['readmitted'].value_counts().items()},
        },
        'by_age': rate_table(df, 'age', age_order),
        'by_prior_inpatient': rate_table(df, 'inpatient_bucket', ['0', '1', '2', '3+']),
        'by_los': rate_table(df, 'los_bucket', ['1-2 d', '3-4 d', '5-7 d', '8+ d']),
        'by_meds': rate_table(df, 'meds_bucket', ['0-10', '11-15', '16-20', '21+']),
        'by_diag1': sorted(rate_table(df, 'diag_1_group'), key=lambda x: -x['rate']),
        'by_a1c': rate_table(df, 'A1Cresult', ['None', 'Norm', '>7', '>8']),
        'by_insulin': rate_table(df, 'insulin', ['No', 'Steady', 'Down', 'Up']),
        'feature_importance': importances,
    }


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RAW_DATASET
    print(f'Loading {csv}')
    df_raw = load_raw(csv)
    assert df_raw.shape == (101766, 50), df_raw.shape
    df = apply_target_and_exclusions(df_raw)
    train_df, val_df, test_df = patient_grouped_split(df)
    print('split sizes', len(train_df), len(val_df), len(test_df))

    X_tr = engineer_features(train_df)
    pre = build_preprocessor(X_tr)
    Xtr = pre.fit_transform(X_tr)
    Xva = pre.transform(engineer_features(val_df))
    Xte = pre.transform(engineer_features(test_df))
    print('encoded features', Xtr.shape[1])

    # EXP-ML-05 hyper-parameters, unchanged.
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                             scale_pos_weight=7.8, random_state=config.SEED, verbose=-1)
    t0 = time.time()
    clf.fit(Xtr, train_df['target'].values)
    print(f'trained in {time.time() - t0:.1f}s')

    val_m = compute_metrics(val_df['target'], clf.predict_proba(Xva)[:, 1])
    test_m = compute_metrics(test_df['target'], clf.predict_proba(Xte)[:, 1])
    print('VAL ', {k: round(v, 4) for k, v in val_m.items() if isinstance(v, float)})
    print('TEST', {k: round(v, 4) for k, v in test_m.items() if isinstance(v, float)})

    config.MODELS_DIR.mkdir(exist_ok=True)
    pipe = Pipeline([('preprocessor', pre), ('classifier', clf)])
    joblib.dump(pipe, config.BASE_MODEL_PATH, compress=3)

    names = pre.get_feature_names_out()
    imp = clf.booster_.feature_importance(importance_type='gain')
    top = np.argsort(imp)[::-1][:15]
    importances = [{'feature': str(names[i]).replace('num__', '').replace('cat__', ''),
                    'gain': round(float(imp[i] / imp.sum() * 100), 2)} for i in top]

    base = {'model': 'LightGBM (Cost-Sensitive) - EXP-ML-05', 'encoded_features': int(Xtr.shape[1]),
            'train_encounters': len(train_df), 'val_encounters': len(val_df),
            'test_encounters': len(test_df), 'validation': val_m, 'test': test_m,
            'hyperparameters': {'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 31,
                                'scale_pos_weight': 7.8, 'random_state': config.SEED}}
    config.BASE_METRICS_PATH.write_text(json.dumps(base, indent=2))

    # ---- static assets -------------------------------------------------
    config.DATA_DIR.mkdir(exist_ok=True)
    config.INSIGHTS_PATH.write_text(json.dumps(build_insights(df_raw, df, importances), indent=2))

    rng = np.random.RandomState(config.SEED)
    ref = test_df.sample(3000, random_state=config.SEED)
    ref[RAW_COLUMNS].to_csv(config.REFERENCE_SAMPLE_PATH, index=False)

    # Known examples (held-out test encounters): 3 readmitted <30d, 3 not readmitted.
    prob = clf.predict_proba(Xte)[:, 1]
    test_df = test_df.assign(_p=prob)
    pos = test_df[test_df.target == 1].sort_values('_p', ascending=False)
    neg = test_df[test_df.target == 0].sort_values('_p')
    picks = pd.concat([pos.head(3), neg.head(3)])   # clear-cut cases for the demo
    examples = []
    for _, r in picks.iterrows():
        rec = {c: (r[c].item() if hasattr(r[c], 'item') else r[c]) for c in RAW_COLUMNS
               if c not in ('encounter_id', 'patient_nbr', 'weight', 'payer_code', 'readmitted')}
        examples.append({'encounter_id': int(r['encounter_id']), 'actual': r['readmitted'],
                         'record': rec})
    config.EXAMPLES_PATH.write_text(json.dumps(examples, indent=2))

    # Upload demo files come from the validation split (test split stays untouched).
    for name, n in (('sample_retrain_small.csv', 1500), ('sample_retrain_large.csv', 6000)):
        val_df.sample(n, random_state=config.SEED)[RAW_COLUMNS].to_csv(config.DATA_DIR / name, index=False)
    ul = val_df.sample(200, random_state=7)[[c for c in RAW_COLUMNS if c != 'readmitted']]
    ul.to_csv(config.DATA_DIR / 'sample_unlabeled.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
