"""Flask application: dashboard, prediction, retraining, evaluation and insights."""
import csv
import io
import json
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from readmission import config, db
from readmission.features import MEDICATION_COLUMNS, RAW_DEFAULTS
from readmission.model_service import service
from readmission.validation import ValidationError, validate_labeled, validate_unlabeled

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('readmission')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES + 1024 * 1024
db.init_db()

AGE_BRACKETS = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                '[60-70)', '[70-80)', '[80-90)', '[90-100)']
ADMISSION_TYPES = {1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Newborn', 5: 'Not available',
                   7: 'Trauma center', 8: 'Not mapped'}
DISCHARGES = {1: 'Discharged to home', 2: 'Short-term hospital', 3: 'Skilled nursing facility',
              4: 'Intermediate care facility', 5: 'Other inpatient care', 6: 'Home with health service',
              7: 'Left against medical advice', 8: 'Home with IV service', 22: 'Rehabilitation facility',
              23: 'Long-term care hospital', 28: 'Psychiatric hospital'}
SOURCES = {1: 'Physician referral', 2: 'Clinic referral', 3: 'HMO referral', 4: 'Transfer from hospital',
           5: 'Transfer from skilled nursing', 6: 'Transfer from other facility', 7: 'Emergency room',
           9: 'Not available', 17: 'Not mapped'}
SPECIALTIES = ['?', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology',
               'Surgery-General', 'Nephrology', 'Orthopedics', 'Orthopedics-Reconstructive',
               'Radiologist', 'Pulmonology', 'Other']
LIMITS = {'time_in_hospital': (1, 14), 'num_lab_procedures': (0, 140), 'num_procedures': (0, 10),
          'num_medications': (0, 100), 'number_outpatient': (0, 50), 'number_emergency': (0, 50),
          'number_inpatient': (0, 30), 'number_diagnoses': (1, 20)}


def read_json(name, default=None):
    p = Path(name)
    return json.loads(p.read_text()) if p.exists() else default


def err(message, status=400, **extra):
    return jsonify({'ok': False, 'error': message, **extra}), status


def clean_form_record(data):
    """Validate a single-patient payload. Returns (record, errors)."""
    rec, errors = {}, {}
    for field, (lo, hi) in LIMITS.items():
        try:
            val = int(float(data.get(field, RAW_DEFAULTS[field])))
        except (TypeError, ValueError):
            errors[field] = 'must be a number'
            continue
        if not lo <= val <= hi:
            errors[field] = f'must be between {lo} and {hi}'
        rec[field] = val
    for field, allowed in (('age', AGE_BRACKETS), ('gender', ['Female', 'Male', 'Unknown/Invalid']),
                           ('A1Cresult', ['None', 'Norm', '>7', '>8']),
                           ('max_glu_serum', ['None', 'Norm', '>200', '>300']),
                           ('change', ['No', 'Ch']), ('diabetesMed', ['Yes', 'No']),
                           ('medical_specialty', SPECIALTIES)):
        val = data.get(field, RAW_DEFAULTS[field])
        if val not in allowed:
            errors[field] = 'invalid choice'
        rec[field] = val
    for field, allowed in (('admission_type_id', ADMISSION_TYPES), ('discharge_disposition_id', DISCHARGES),
                           ('admission_source_id', SOURCES)):
        try:
            val = int(data.get(field, RAW_DEFAULTS[field]))
        except (TypeError, ValueError):
            val = None
        if val is None or (val not in allowed and not 1 <= val <= 40) \
                or (field == 'discharge_disposition_id' and val in config.EXCLUDED_DISCHARGE_IDS):
            errors[field] = 'invalid choice'
        rec[field] = val
    for field in ('diag_1', 'diag_2', 'diag_3'):
        val = str(data.get(field, '') or '?').strip()[:10]
        rec[field] = val if val else '?'
    for field in ('race',):
        rec[field] = str(data.get(field, RAW_DEFAULTS[field]))[:30]
    for med in MEDICATION_COLUMNS:
        val = data.get(med, 'No')
        if val not in ('No', 'Steady', 'Up', 'Down'):
            errors[med] = 'invalid choice'
        rec[med] = val
    return rec, errors


# ---- pages -------------------------------------------------------------------------------
@app.route('/')
def dashboard():
    base = read_json(config.BASE_METRICS_PATH, {})
    return render_template('dashboard.html', page='dashboard', active=service.active(), stats=db.stats(),
                           base=base, recent=db.recent_predictions(6),
                           insights=read_json(config.INSIGHTS_PATH, {}))


@app.route('/predict')
def predict_page():
    return render_template('predict.html', page='predict', age_brackets=AGE_BRACKETS,
                           admission_types=ADMISSION_TYPES, discharges=DISCHARGES, sources=SOURCES,
                           specialties=SPECIALTIES, defaults=RAW_DEFAULTS, active=service.active())


@app.route('/retrain')
def retrain_page():
    return render_template('retrain.html', page='retrain', uploads=db.list_uploads(),
                           versions=db.list_versions(), active=service.active(),
                           stats=db.stats(), ft_trees=config.FT_ADDITIONAL_TREES,
                           ft_lr=config.FT_LEARNING_RATE, min_rows=config.MIN_UPLOAD_ROWS)


@app.route('/model')
def model_page():
    base = read_json(config.BASE_METRICS_PATH, {})
    experiments = pd.read_csv(config.RESULTS_DIR / 'experiment_results.csv').to_dict('records') \
        if (config.RESULTS_DIR / 'experiment_results.csv').exists() else []
    final = pd.read_csv(config.RESULTS_DIR / 'final_test_evaluation.csv').to_dict('records') \
        if (config.RESULTS_DIR / 'final_test_evaluation.csv').exists() else []
    return render_template('model.html', page='model', base=base, experiments=experiments, final=final,
                           versions=db.list_versions(), active=service.active(),
                           insights=read_json(config.INSIGHTS_PATH, {}))


@app.route('/insights')
def insights_page():
    return render_template('insights.html', page='insights', active=service.active())


@app.route('/results/<path:filename>')
def results_file(filename):
    return send_from_directory(config.RESULTS_DIR, filename)


@app.route('/samples/<path:filename>')
def sample_file(filename):
    if not filename.startswith('sample_') or not filename.endswith('.csv'):
        return err('Not found', 404)
    return send_from_directory(config.DATA_DIR, filename, as_attachment=True)


# ---- JSON API -----------------------------------------------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': service.active()['name']})


@app.route('/api/insights')
def api_insights():
    return jsonify(read_json(config.INSIGHTS_PATH, {}))


@app.route('/api/examples')
def api_examples():
    return jsonify(read_json(config.EXAMPLES_PATH, []))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return err('Send the patient fields as a JSON object.')
    rec, errors = clean_form_record(data)
    if errors:
        return err('Some inputs are invalid.', fields=errors)
    try:
        result = service.predict_records([rec], explain=True)[0]
    except Exception:  # noqa: BLE001
        log.exception('prediction failed')
        return err('Prediction failed on the server.', 500)
    db.log_predictions([('form', rec, result['probability'], result['label'], result['model_version_id'])])
    return jsonify({'ok': True, 'result': result})


@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    f = request.files.get('file')
    if not f or not f.filename:
        return err('Choose a CSV file to upload.')
    try:
        df, report = validate_unlabeled(f.read())
        records = df.to_dict('records')
        results = service.predict_records(records)
    except ValidationError as exc:
        return err(str(exc))
    except Exception:  # noqa: BLE001
        log.exception('batch prediction failed')
        return err('Batch prediction failed on the server.', 500)
    db.log_predictions([('batch', r, p['probability'], p['label'], p['model_version_id'])
                        for r, p in zip(records, results)])
    rows = [{'row': i + 1, 'encounter_id': r.get('encounter_id', ''), 'probability': round(p['probability'], 4),
             'label': p['label'], 'risk_band': p['risk_band']}
            for i, (r, p) in enumerate(zip(records, results))]
    summary = {'rows': len(rows), 'flagged_high_risk': sum(r['label'] for r in rows),
               'mean_probability': round(sum(r['probability'] for r in rows) / len(rows), 4)}
    return jsonify({'ok': True, 'summary': summary, 'report': report, 'rows': rows,
                    'model_name': results[0]['model_name']})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    f = request.files.get('file')
    if not f or not f.filename:
        return err('Choose a CSV file to upload.')
    try:
        df, report = validate_labeled(f.read(), db.existing_encounter_ids())
        uid = db.save_upload(f.filename, df, report)
    except ValidationError as exc:
        return err(str(exc))
    except Exception:  # noqa: BLE001
        log.exception('upload failed')
        return err('Upload failed on the server.', 500)
    log.info('stored upload %s (%s rows)', uid, len(df))
    return jsonify({'ok': True, 'upload_id': uid, 'report': report,
                    'total_records': db.count_training_records()})


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    data = request.get_json(silent=True) or {}
    try:
        trees = int(data.get('additional_trees', config.FT_ADDITIONAL_TREES))
        lr = float(data.get('learning_rate', config.FT_LEARNING_RATE))
        if not (1 <= trees <= 500) or not (0.001 <= lr <= 0.3):
            return err('additional_trees must be 1-500 and learning_rate 0.001-0.3.')
        ids = data.get('upload_ids') or None
        result = service.retrain(upload_ids=[int(i) for i in ids] if ids else None, extra_trees=trees,
                                 learning_rate=lr, activate=bool(data.get('activate', True)))
    except (ValueError, TypeError) as exc:
        return err(str(exc))
    except Exception:  # noqa: BLE001
        log.exception('retraining failed')
        return err('Retraining failed on the server.', 500)
    return jsonify({'ok': True, **result})


@app.route('/api/versions/<int:version_id>/activate', methods=['POST'])
def api_activate(version_id):
    try:
        db.activate_version(version_id)
    except KeyError:
        return err('Unknown model version.', 404)
    return jsonify({'ok': True, 'active': service.active()})


@app.errorhandler(413)
def too_large(_):
    return err('File too large.', 413)


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith('/api/'):
        return err('Not found', 404)
    return render_template('error.html', page='', active=service.active(), code=404,
                           message='That page does not exist.'), 404


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
