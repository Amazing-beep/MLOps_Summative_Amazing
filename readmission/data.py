"""Dataset loading, target definition and patient-grouped splitting (from the notebook)."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import EXCLUDED_DISCHARGE_IDS, SEED


def load_raw(path):
    # keep_default_na=False preserves the literal 'None' lab-test marker.
    return pd.read_csv(path, keep_default_na=False)


def apply_target_and_exclusions(df):
    """target = 1 if readmitted < 30 days; drop in-hospital deaths."""
    out = df.copy()
    out['discharge_disposition_id'] = out['discharge_disposition_id'].astype(int)
    out = out[~out['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_IDS)].copy()
    out['target'] = (out['readmitted'] == '<30').astype(int)
    return out.reset_index(drop=True)


def patient_grouped_split(df, test_size=0.15, val_size=0.15, random_state=SEED):
    """70/15/15 split grouped by patient_nbr (zero patient overlap)."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, te_idx = next(g1.split(df, df['target'], groups=df['patient_nbr']))
    tv = df.iloc[tv_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    g2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state)
    tr_idx, va_idx = next(g2.split(tv, tv['target'], groups=tv['patient_nbr']))
    return (tv.iloc[tr_idx].reset_index(drop=True),
            tv.iloc[va_idx].reset_index(drop=True), test)
