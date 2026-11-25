"""Central paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
# Writable runtime state (SQLite DB + retrained model versions).
RUNTIME_DIR = Path(os.environ.get("RUNTIME_DIR", ROOT / "runtime"))
DB_PATH = Path(os.environ.get("DB_PATH", RUNTIME_DIR / "readmission.db"))
VERSIONS_DIR = RUNTIME_DIR / "model_versions"

RAW_DATASET = DATA_DIR / "diabetic_data.csv"
BASE_MODEL_PATH = MODELS_DIR / "best_traditional_model.joblib"
BASE_METRICS_PATH = MODELS_DIR / "base_metrics.json"
ORIGINAL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
INSIGHTS_PATH = DATA_DIR / "insights.json"
EXAMPLES_PATH = DATA_DIR / "examples.json"
REFERENCE_SAMPLE_PATH = DATA_DIR / "reference_test_sample.csv"
RESULTS_DIR = ROOT / "results"

SEED = 42
DECISION_THRESHOLD = 0.5          # same operating point as the original notebook
EXCLUDED_DISCHARGE_IDS = [11, 19, 20, 21]  # expired in hospital (original exclusion)

# Fine-tuning defaults (continued boosting from the pre-trained LightGBM).
FT_ADDITIONAL_TREES = 100         # maximum extra boosting rounds (early stopping may use fewer)
FT_LEARNING_RATE = 0.01
FT_EARLY_STOPPING_ROUNDS = 10
# Conservative regularisation so small uploads cannot wipe out the pre-trained knowledge.
FT_TREE_PARAMS = {'num_leaves': 8, 'min_child_samples': 50, 'reg_lambda': 5.0}
MIN_UPLOAD_ROWS = 100
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
