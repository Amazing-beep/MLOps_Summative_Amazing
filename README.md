# Hospital Readmission Risk — MLOps Deploymen

A web application that predicts whether a patient with diabetes will be **readmitted to hospital within 30 days of discharge**, and lets users **upload newly labelled data and fine-tune the model** without leaving the browser.

It operationalises the project from my *Introduction to Machine Learning* summative
(<https://github.com/Amazing-beep/diabetes-readmission-ml-dl>): the same dataset, prediction task, preprocessing and model.

| | |
|---|---|
| **Task** | Binary classification — `readmitted == '<30'` vs. everything else |
| **Dataset** | UCI *Diabetes 130-US Hospitals (1999–2008)*, 101,766 encounters × 50 columns, CC BY 4.0 |
| **Deployed model** | Cost-sensitive LightGBM (EXP-ML-05: 150 trees, lr 0.05, 31 leaves, `scale_pos_weight=7.8`) inside a scikit-learn pipeline with the original `ColumnTransformer` (206 encoded features) |
| **Held-out test results** | ROC-AUC 0.6566 · PR-AUC 0.2040 · Recall 0.5417 · Precision 0.1781 · F1 0.2680 · Accuracy 0.6687 (15,000 encounters, patient-grouped split) |

> **Where the model comes from.** The original repository did not commit the dataset or model binaries (they were on Google Drive). `scripts/train_baseline.py` re-runs the notebook's exact pipeline (same split, seed and hyper-parameters) and reproduces the reported test metrics to four decimals; this is asserted in `tests/test_app.py`. The resulting `models/best_traditional_model.joblib` (~0.5 MB) is committed so the app runs out of the box.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Active model, original test metrics, database counts, recent predictions |
| **Predict** | Form (or one-click load of a *real* dataset encounter) → probability, class, risk band, and the top factors driving the prediction (LightGBM TreeSHAP contributions). Also **batch prediction** from a CSV with downloadable results |
| **Upload & Retrain** | Upload a labelled CSV → validation → saved to SQLite → fine-tune → before/after comparison → activate or roll back model versions |
| **Model Evaluation** | Original notebook results: 10 experiments, test table, confusion matrices, ROC/PR and learning curves, interpretation, version history |
| **Data Insights** | Readmission rates by prior inpatient stays, diagnosis, age, length of stay, medications, HbA1c, insulin; class balance; model feature importance |

### Retraining pipeline

1. **Validate** (`readmission/validation.py`): required columns, numeric/range checks, valid age brackets and labels, removal of in-hospital deaths (original exclusion rule), duplicate `encounter_id`s (within file and against the database), minimum size and class counts. Errors are explained in the UI.
2. **Persist** (`readmission/db.py`): rows go to the SQLite table `training_records`, with an `uploads` audit table.
3. **Preprocess**: the *same* `engineer_features` + fitted `ColumnTransformer` as the original model (`readmission/features.py` — copied unchanged from the notebook).
4. **Fine-tune** (`readmission/model_service.py`): LightGBM `init_model` continues boosting from the currently active booster on 80% of the stored data (patient-grouped), with small trees, L2 regularisation, a low learning rate and **early stopping** on the other 20%. This is conservative on purpose: small uploads should refine the model, not overwrite it.
5. **Evaluate**: old vs. new model on (a) the held-out part of the new data and (b) a fixed 3,000-row sample of the original test split, so regressions are visible (7 metrics shown).
6. **Version & serve**: the new pipeline is saved under `runtime/model_versions/`, registered in `model_versions`, and — if activated — used by the Predict page immediately. Any version can be re-activated (rollback).

Honest note: the sample upload files come from the same distribution as the original data, so fine-tuning changes the model only slightly (e.g. 150 → ~157 trees). The pipeline is built for genuinely new/drifted data; the UI reports the before/after metrics rather than claiming an improvement.

## Architecture

```
Browser ──► Flask (app.py) ──► readmission/
  Jinja pages + vanilla JS        ├─ features.py       original feature engineering + ColumnTransformer
  /api/predict, /api/upload,      ├─ validation.py     CSV validation
  /api/retrain, ...               ├─ model_service.py  load / predict / explain / fine-tune / evaluate
                                  ├─ db.py             SQLite (uploads, training_records, model_versions, predictions)
                                  └─ metrics.py        ROC/PR-AUC, recall, precision, F1, accuracy, Brier, log-loss
models/   original pipeline (v1) + metrics         runtime/  SQLite DB + retrained versions (git-ignored)
notebooks/ original model-development notebook      results/  original experiment tables & figures
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # http://localhost:5000
```

Production-style (same as Docker): `gunicorn app:app --bind 0.0.0.0:8000 --workers 1 --threads 4`

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python -m pytest -q
python app.py                        # http://localhost:5000
```

### Docker

```bash
docker build -t readmission-risk .
docker run -p 8000:8000 readmission-risk               # http://localhost:8000
# keep the database/model versions across restarts:
docker run -p 8000:8000 -v readmission-data:/app/runtime readmission-risk
```

### Tests

```bash
python -m pytest -q          # 13 tests: pages, known-example predictions, validation, DB, retraining, rollback
```

### Rebuild the baseline model (optional)

Download `diabetic_data.csv` into `data/` (see `data/README.md`), then `python scripts/train_baseline.py`. It regenerates the model, metrics, insights, examples and sample files.

### Deploy publicly (Render, free tier)

`render.yaml` is included. Create a Render account → *New → Blueprint* → select this repository → deploy. SQLite lives on the container's ephemeral disk on the free plan: uploads and retrained versions reset on redeploy (attach a persistent disk mounted at `/app/runtime` to keep them).

## The notebook

`notebooks/diabetes_readmission_classification.ipynb` is the original, executed notebook (outputs preserved): data loading and audit, leakage-safe patient-grouped train/validation/test split, preprocessing, 6 traditional + 4 deep-learning experiments (regularisation, dropout, early stopping, learning-rate/width variations, cost-sensitive weighting), ≥7 metrics (ROC-AUC, PR-AUC, recall, precision, F1, accuracy, specificity, Brier), learning curves, ROC/PR curves, confusion matrices, TreeSHAP interpretation and error analysis. To re-run it: `pip install -r requirements-notebook.txt` and place the dataset in `data/` (it can also download it from UCI automatically).

## Limitations

Historical data (1999–2008); ROC-AUC ≈ 0.66 means the model is a screening aid, not a diagnostic tool; threshold 0.5 favours recall (≈54%) over precision (≈18%). Not for clinical use.

## Credits

Dataset: Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research International — UCI ML Repository, CC BY 4.0.
