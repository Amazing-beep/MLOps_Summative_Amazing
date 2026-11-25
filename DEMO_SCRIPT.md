# Video demonstration script (target 5–6 min, camera ON)

Setup before recording: `python app.py`, open http://localhost:5000, have `data/sample_retrain_small.csv` ready, and a fresh DB (delete the `runtime/` folder). Keep your camera visible while you speak and share your screen.

**0:00 – Intro (20 s, face to camera)** "I'm <name>. This app deploys my Introduction-to-ML model, which predicts 30-day readmission for diabetic patients using the UCI Diabetes 130-US Hospitals data. It has prediction, data upload, retraining, evaluation and data insights."

**0:20 – Dashboard (30 s)** Point out the active model v1, the original test ROC-AUC 0.6566, the empty database counters, and the nav bar.

**0:50 – Prediction with new data (90 s)** Go to *Predict*.
1. Load the first real example → *Predict*. Show ~87% probability, "readmitted < 30 days", the key factors (prior inpatient visits), and "model agrees ✓" with the recorded outcome.
2. Load a "not readmitted" example → ~5%, agrees ✓.
3. Edit values by hand (e.g. set *Inpatient visits* to 6, *Days in hospital* 8) → predict again; show the probability change.
4. Type 99 days in hospital → show the validation message (error handling).
5. Batch: upload `sample_unlabeled.csv` → show the summary and the download button.

**2:30 – Upload new data (60 s)** Go to *Upload & Retrain*. First upload a wrong file to show the clear validation error. Then upload `sample_retrain_small.csv` → *Validate & save*. Show the summary (1,500 rows stored, 174 positive) and the new row in the "Uploaded datasets (database)" table. Mention SQLite.

**3:30 – Retraining (90 s)** Keep the defaults, click *Retrain model*; show the loading state. Explain: fine-tunes the pre-trained LightGBM with LightGBM `init_model` (continues boosting), same preprocessing, early stopping on a held-out 20%. Walk through the before/after tables (new data and fixed reference set) and the trees 150 → ~157. Show the new version v2 is active in the version table; go back to *Predict* and predict again — the model badge shows v2. Optionally click *Activate* on v1 to show rollback.

**5:00 – Evaluation & insights (45 s)** *Model Evaluation*: metrics (ROC-AUC, PR-AUC, recall, precision, F1, accuracy), the 10-experiment table, confusion matrix and learning curves; mention the notebook. *Data Insights*: prior inpatient stays are the strongest signal (8.6% → 26.2%).

**5:45 – Close (15 s)** "Everything is in the GitHub repo with a Dockerfile and tests."
