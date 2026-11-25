"""Loading, prediction and fine-tuning of the readmission model."""
import threading
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from . import config, db
from .data import apply_target_and_exclusions
from .features import NUMERIC_RAW, RAW_COLUMNS, complete_record, engineer_features
from .metrics import compute_metrics


def _risk_band(p):
    if p >= 0.65:
        return 'Very high'
    if p >= config.DECISION_THRESHOLD:
        return 'High'
    if p >= 0.35:
        return 'Moderate'
    return 'Lower'


class ModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipe = None
        self._version_id = None
        self._ref = None

    # ---- loading ----------------------------------------------------------
    def _load(self):
        v = db.get_active_version()
        if v['id'] != self._version_id:
            with self._lock:
                self._pipe = joblib.load(v['path'])
                self._version_id = v['id']
                self._version = v
        return self._pipe, self._version

    def active(self):
        return self._load()[1]

    # ---- prediction -----------------------------------------------------------
    @staticmethod
    def _frame(records):
        return pd.DataFrame([complete_record(r) for r in records])

    def predict_records(self, records, explain=False):
        pipe, version = self._load()
        raw = self._frame(records)
        X_eng = engineer_features(raw)
        X = pipe.named_steps['preprocessor'].transform(X_eng)
        clf = pipe.named_steps['classifier']
        prob = clf.predict_proba(X)[:, 1]
        out = []
        for i, p in enumerate(prob):
            item = {'probability': float(p), 'label': int(p >= config.DECISION_THRESHOLD),
                    'risk_band': _risk_band(p), 'threshold': config.DECISION_THRESHOLD,
                    'model_version_id': version['id'], 'model_name': version['name']}
            out.append(item)
        if explain:
            contrib = clf.predict(X, pred_contrib=True)[:, :-1]
            names = pipe.named_steps['preprocessor'].get_feature_names_out()
            for i, item in enumerate(out):
                item['factors'] = self._factors(contrib[i], names, X_eng.iloc[i])
        return out

    @staticmethod
    def _factors(contrib, names, eng_row, top=6):
        order = np.argsort(np.abs(contrib))[::-1][:top]
        res = []
        for j in order:
            if abs(contrib[j]) < 1e-4:
                continue
            n = str(names[j])
            if n.startswith('num__'):
                col = n[5:]
                label = f'{col} = {eng_row[col]:g}' if col in eng_row else col
            else:
                n = n[5:]
                label = n
                for col in eng_row.index:
                    if n.startswith(col + '_'):
                        label = f'{col} = {n[len(col) + 1:]}'
                        break
            res.append({'feature': label, 'effect': float(contrib[j]),
                        'direction': 'raises risk' if contrib[j] > 0 else 'lowers risk'})
        return res

    # ---- evaluation ---------------------------------------------------------------
    def reference_frame(self):
        if self._ref is None:
            self._ref = pd.read_csv(config.REFERENCE_SAMPLE_PATH, keep_default_na=False)
            for c in NUMERIC_RAW:
                self._ref[c] = self._ref[c].astype(int)
        return self._ref

    @staticmethod
    def _score(pipe, raw_df, y):
        X = pipe.named_steps['preprocessor'].transform(engineer_features(raw_df))
        return compute_metrics(y, pipe.named_steps['classifier'].predict_proba(X)[:, 1])

    # ---- retraining ----------------------------------------------------------------
    def retrain(self, upload_ids=None, extra_trees=config.FT_ADDITIONAL_TREES,
                learning_rate=config.FT_LEARNING_RATE, activate=True):
        """Fine-tune the active model on the stored uploaded data.

        The fitted ColumnTransformer of the parent model is reused unchanged (same
        feature space as the original model). Additional boosting rounds are added
        on top of the parent booster via LightGBM ``init_model``.
        """
        t0 = time.time()
        parent_pipe, parent = self._load()
        df = db.load_training_frame(upload_ids)
        if df.empty:
            raise ValueError('No training data in the database yet. Upload a labelled dataset first.')
        for c in NUMERIC_RAW:
            df[c] = df[c].astype(int)
        df = apply_target_and_exclusions(df)
        y = df['target'].values
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            raise ValueError('Need at least 10 positive and 10 negative rows to retrain.')

        # Hold out 20 % for validation (grouped by patient when ids are present).
        groups = df['patient_nbr'].where(df['patient_nbr'] != '', df.index.astype(str))
        if groups.nunique() < len(df):
            tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=config.SEED)
                          .split(df, y, groups=groups))
        else:
            tr, va = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y,
                                      random_state=config.SEED)
        tr_df, va_df = df.iloc[tr], df.iloc[va]
        if tr_df['target'].nunique() < 2 or va_df['target'].nunique() < 2:
            raise ValueError('Both classes must appear in the training and validation parts.')

        pre = parent_pipe.named_steps['preprocessor']
        parent_clf = parent_pipe.named_steps['classifier']
        X_tr = pre.transform(engineer_features(tr_df))
        X_va = pre.transform(engineer_features(va_df))

        params = dict(parent_clf.get_params())
        params.update(n_estimators=int(extra_trees), learning_rate=float(learning_rate), **config.FT_TREE_PARAMS)
        new_clf = lgb.LGBMClassifier(**params)
        new_clf.fit(X_tr, tr_df['target'].values,
                    eval_set=[(X_va, va_df['target'].values)], eval_metric='auc',
                    init_model=parent_clf.booster_,
                    callbacks=[lgb.early_stopping(config.FT_EARLY_STOPPING_ROUNDS, verbose=False)])
        child = Pipeline([('preprocessor', pre), ('classifier', new_clf)])

        # Before/after comparison on (a) held-out part of the new data, (b) a fixed
        # reference sample from the original held-out test split.
        ref = self.reference_frame()
        metrics = {
            'new_data_validation': {'before': self._score(parent_pipe, va_df, va_df['target']),
                                    'after': self._score(child, va_df, va_df['target'])},
            'reference_test': {'before': self._score(parent_pipe, ref, (ref['readmitted'] == '<30').astype(int)),
                               'after': self._score(child, ref, (ref['readmitted'] == '<30').astype(int))},
            'trees_before': int(parent_clf.booster_.num_trees()),
            'trees_after': int(new_clf.best_iteration_ or new_clf.booster_.num_trees()),
            'duration_sec': round(time.time() - t0, 2),
        }

        vnum = db.next_version_number()
        config.VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.VERSIONS_DIR / f'model_v{vnum}.joblib'
        joblib.dump(child, path, compress=3)
        params_out = {'additional_trees': int(extra_trees), 'learning_rate': float(learning_rate),
                      'early_stopping_rounds': config.FT_EARLY_STOPPING_ROUNDS, **config.FT_TREE_PARAMS,
                      'parent_version_id': parent['id']}
        vid = db.add_version(f'v{vnum} - fine-tuned from v{parent["id"]}', path, parent['id'],
                             len(tr_df), len(va_df), upload_ids or 'all', params_out, metrics, activate)
        return {'version_id': vid, 'name': f'v{vnum} - fine-tuned from v{parent["id"]}',
                'activated': activate, 'n_train': len(tr_df), 'n_val': len(va_df),
                'metrics': metrics, 'params': params_out}


service = ModelService()
