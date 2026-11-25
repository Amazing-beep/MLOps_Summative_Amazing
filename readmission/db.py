"""SQLite persistence: uploads, training records, model versions, predictions."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from . import config
from .features import RAW_COLUMNS


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def _q(col):
    return '"' + col.replace('"', '""') + '"'


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    cols = ',\n'.join(f'{_q(c)} TEXT' for c in RAW_COLUMNS)
    with connect() as c:
        c.executescript(f"""
        CREATE TABLE IF NOT EXISTS uploads(
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, uploaded_at TEXT,
            rows_received INTEGER, rows_stored INTEGER, rows_rejected INTEGER,
            rows_duplicate INTEGER, positives INTEGER, report_json TEXT);
        CREATE TABLE IF NOT EXISTS training_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL REFERENCES uploads(id), {cols});
        CREATE INDEX IF NOT EXISTS idx_tr_upload ON training_records(upload_id);
        CREATE TABLE IF NOT EXISTS model_versions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TEXT,
            path TEXT, parent_id INTEGER, is_active INTEGER DEFAULT 0,
            kind TEXT, n_train INTEGER, n_val INTEGER, upload_ids TEXT,
            params_json TEXT, metrics_json TEXT);
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT,
            input_json TEXT, probability REAL, label INTEGER, model_version_id INTEGER);
        """)
        if c.execute('SELECT COUNT(*) FROM model_versions').fetchone()[0] == 0:
            metrics = {}
            if config.BASE_METRICS_PATH.exists():
                metrics = json.loads(config.BASE_METRICS_PATH.read_text())
            c.execute(
                'INSERT INTO model_versions(name,created_at,path,is_active,kind,metrics_json) '
                'VALUES(?,?,?,?,?,?)',
                ('v1 - original LightGBM (Intro to ML)', now(), str(config.BASE_MODEL_PATH), 1,
                 'base', json.dumps({'test': metrics.get('test'), 'validation': metrics.get('validation')})))


# ---- uploads / training data ------------------------------------------------
def existing_encounter_ids():
    with connect() as c:
        return {r[0] for r in c.execute(
            "SELECT DISTINCT encounter_id FROM training_records WHERE encounter_id != ''")}


def save_upload(filename, df, report):
    """Persist a validated labelled dataframe. Returns upload id."""
    out = df.reindex(columns=RAW_COLUMNS).fillna('').astype(str)
    with connect() as c:
        cur = c.execute(
            'INSERT INTO uploads(filename,uploaded_at,rows_received,rows_stored,rows_rejected,'
            'rows_duplicate,positives,report_json) VALUES(?,?,?,?,?,?,?,?)',
            (filename, now(), report['rows_received'], len(out), report['rows_rejected'],
             report['rows_duplicate'], int((out['readmitted'] == '<30').sum()), json.dumps(report)))
        uid = cur.lastrowid
        cols = ['upload_id'] + RAW_COLUMNS
        sql = f"INSERT INTO training_records({','.join(_q(x) for x in cols)}) VALUES({','.join('?' * len(cols))})"
        c.executemany(sql, [(uid, *row) for row in out.itertuples(index=False, name=None)])
    return uid


def list_uploads():
    with connect() as c:
        return [dict(r) for r in c.execute('SELECT * FROM uploads ORDER BY id DESC')]


def load_training_frame(upload_ids=None):
    q = 'SELECT * FROM training_records'
    args = []
    if upload_ids:
        q += f" WHERE upload_id IN ({','.join('?' * len(upload_ids))})"
        args = list(upload_ids)
    with connect() as c:
        df = pd.read_sql_query(q, c, params=args)
    return df.drop(columns=['id', 'upload_id'])


def count_training_records():
    with connect() as c:
        return c.execute('SELECT COUNT(*) FROM training_records').fetchone()[0]


# ---- model versions -----------------------------------------------------------
def _version_row(r):
    d = dict(r)
    d['metrics'] = json.loads(d.pop('metrics_json') or '{}')
    d['params'] = json.loads(d.pop('params_json') or '{}')
    return d


def list_versions():
    with connect() as c:
        return [_version_row(r) for r in c.execute('SELECT * FROM model_versions ORDER BY id DESC')]


def get_active_version():
    with connect() as c:
        r = c.execute('SELECT * FROM model_versions WHERE is_active=1 ORDER BY id DESC LIMIT 1').fetchone()
    return _version_row(r)


def add_version(name, path, parent_id, n_train, n_val, upload_ids, params, metrics, activate):
    with connect() as c:
        if activate:
            c.execute('UPDATE model_versions SET is_active=0')
        cur = c.execute(
            'INSERT INTO model_versions(name,created_at,path,parent_id,is_active,kind,n_train,n_val,'
            'upload_ids,params_json,metrics_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (name, now(), str(path), parent_id, 1 if activate else 0, 'fine-tuned', n_train, n_val,
             json.dumps(upload_ids), json.dumps(params), json.dumps(metrics)))
        return cur.lastrowid


def next_version_number():
    with connect() as c:
        return c.execute('SELECT COALESCE(MAX(id),0)+1 FROM model_versions').fetchone()[0]


def activate_version(version_id):
    with connect() as c:
        if not c.execute('SELECT 1 FROM model_versions WHERE id=?', (version_id,)).fetchone():
            raise KeyError(version_id)
        c.execute('UPDATE model_versions SET is_active=0')
        c.execute('UPDATE model_versions SET is_active=1 WHERE id=?', (version_id,))


# ---- predictions ----------------------------------------------------------------
def log_predictions(rows):
    """rows: iterable of (source, input_dict, probability, label, version_id)."""
    with connect() as c:
        c.executemany(
            'INSERT INTO predictions(created_at,source,input_json,probability,label,model_version_id) '
            'VALUES(?,?,?,?,?,?)',
            [(now(), s, json.dumps(i), float(p), int(l), v) for s, i, p, l, v in rows])


def recent_predictions(limit=10):
    with connect() as c:
        return [dict(r) for r in c.execute(
            'SELECT id,created_at,source,probability,label,model_version_id FROM predictions '
            'ORDER BY id DESC LIMIT ?', (limit,))]


def stats():
    with connect() as c:
        return {
            'uploads': c.execute('SELECT COUNT(*) FROM uploads').fetchone()[0],
            'training_records': c.execute('SELECT COUNT(*) FROM training_records').fetchone()[0],
            'predictions': c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],
            'versions': c.execute('SELECT COUNT(*) FROM model_versions').fetchone()[0],
        }
