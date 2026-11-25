"""End-to-end tests: prediction on known examples, validation, DB persistence, retraining."""
import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNTIME_DIR', str(tmp_path))
    for m in [m for m in sys.modules if m == 'app' or m.startswith('readmission')]:
        del sys.modules[m]
    app_module = importlib.import_module('app')
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client(), app_module


EXAMPLES = json.loads((ROOT / 'data' / 'examples.json').read_text())


def test_pages_render(client):
    c, _ = client
    for url in ['/', '/predict', '/retrain', '/model', '/insights', '/health']:
        assert c.get(url).status_code == 200, url
    assert c.get('/nope').status_code == 404


@pytest.mark.parametrize('ex', EXAMPLES, ids=[str(e['encounter_id']) for e in EXAMPLES])
def test_known_examples_predicted_correctly(client, ex):
    """Real held-out encounters: the model must agree with the recorded outcome."""
    c, _ = client
    r = c.post('/api/predict', json=ex['record']).get_json()
    assert r['ok']
    positive = ex['actual'] == '<30'
    assert r['result']['label'] == int(positive)
    assert (r['result']['probability'] >= 0.5) == positive
    assert r['result']['factors']


def test_prediction_matches_original_model_directly(client):
    """The service must return exactly what the original pipeline returns."""
    import joblib
    import pandas as pd
    from readmission import config
    from readmission.features import complete_record, engineer_features
    _, mod = client
    pipe = joblib.load(config.BASE_MODEL_PATH)
    rec = EXAMPLES[0]['record']
    expected = pipe.named_steps["classifier"].predict_proba(
            pipe.named_steps['preprocessor'].transform(engineer_features(pd.DataFrame([complete_record(rec)]))))[:, 1]
    got = mod.service.predict_records([rec])[0]['probability']
    assert abs(got - float(expected[0])) < 1e-9


def test_reproduces_reported_test_metrics():
    m = json.loads((ROOT / 'models' / 'base_metrics.json').read_text())['test']
    assert round(m['roc_auc'], 4) == 0.6566 and round(m['recall'], 4) == 0.5417
    assert round(m['f1'], 4) == 0.268 and round(m['accuracy'], 4) == 0.6687


def test_predict_rejects_bad_input(client):
    c, _ = client
    bad = dict(EXAMPLES[0]['record'], time_in_hospital=99, age='old')
    r = c.post('/api/predict', json=bad)
    assert r.status_code == 400 and 'time_in_hospital' in r.get_json()['fields']
    assert c.post('/api/predict', data='x').status_code == 400


def test_upload_validation_errors(client):
    c, _ = client
    r = c.post('/api/upload', data={'file': (io.BytesIO(b'a,b\n1,2\n'), 'bad.csv')})
    assert r.status_code == 400 and 'Missing' in r.get_json()['error']
    r = c.post('/api/upload', data={'file': (io.BytesIO(b''), 'empty.csv')})
    assert r.status_code == 400
    assert c.post('/api/upload').status_code == 400
    assert c.post('/api/retrain', json={}).status_code == 400  # nothing stored yet


def test_upload_retrain_predict_end_to_end(client):
    c, mod = client
    from readmission import db
    sample = (ROOT / 'data' / 'sample_retrain_small.csv').read_bytes()
    up = c.post('/api/upload', data={'file': (io.BytesIO(sample), 'sample.csv')}).get_json()
    assert up['ok'] and up['report']['rows_stored'] == 1500
    assert db.count_training_records() == 1500          # really persisted in SQLite
    again = c.post('/api/upload', data={'file': (io.BytesIO(sample), 'sample.csv')})
    assert again.status_code == 400                      # all rows duplicates -> rejected

    before = c.get('/health').get_json()['model']
    rt = c.post('/api/retrain', json={'additional_trees': 50, 'learning_rate': 0.02}).get_json()
    assert rt['ok'] and rt['activated']
    assert rt['metrics']['trees_after'] >= rt['metrics']['trees_before']
    assert Path(db.get_active_version()['path']).exists()
    assert c.get('/health').get_json()['model'] != before   # prediction now uses new model

    p = c.post('/api/predict', json=EXAMPLES[0]['record']).get_json()['result']
    assert p['model_version_id'] == rt['version_id']
    # rollback
    assert c.post('/api/versions/1/activate').get_json()['ok']
    assert c.post('/api/predict', json=EXAMPLES[0]['record']).get_json()['result']['model_version_id'] == 1
    assert c.post('/api/versions/99/activate').status_code == 404


def test_batch_prediction(client):
    c, _ = client
    data = (ROOT / 'data' / 'sample_unlabeled.csv').read_bytes()
    r = c.post('/api/predict-batch', data={'file': (io.BytesIO(data), 'u.csv')}).get_json()
    assert r['ok'] and r['summary']['rows'] == 200
    assert all(0 <= x['probability'] <= 1 for x in r['rows'])
