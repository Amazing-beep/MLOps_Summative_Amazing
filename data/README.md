# Data

The full raw dataset (`diabetic_data.csv`, 101,766 × 50, ~19 MB) is **not committed**.
Download it from the UCI repository (CC BY 4.0):
<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
and place `diabetic_data.csv` in this folder. It is only needed to re-run
`scripts/train_baseline.py` or the notebook; the app itself runs without it.

Committed files (all derived from the real dataset):

| File | Purpose |
|---|---|
| `insights.json` | Aggregates for the Data Insights page |
| `examples.json` | 6 real held-out test encounters (3 readmitted <30d, 3 not) for the prediction demo |
| `reference_test_sample.csv` | 3,000 labelled encounters from the original test split; used to compare a model before/after retraining |
| `sample_retrain_small.csv`, `sample_retrain_large.csv` | Labelled files (from the validation split) to demo uploading + retraining |
| `sample_unlabeled.csv` | 200 unlabeled encounters for batch prediction |
