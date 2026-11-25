"""Validation of uploaded CSV files (labelled for retraining, unlabeled for prediction)."""
import io

import pandas as pd

from . import config
from .features import AGE_MAP, MODEL_INPUT_COLUMNS, NUMERIC_RAW, RAW_COLUMNS, REQUIRED_UPLOAD_COLUMNS

VALID_LABELS = {'<30', '>30', 'NO'}
MIN_CLASS_ROWS = 10


class ValidationError(ValueError):
    """Raised with a user-readable message when a file cannot be used."""


def _read_csv(raw_bytes):
    if not raw_bytes:
        raise ValidationError('The uploaded file is empty.')
    if len(raw_bytes) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(f'File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.')
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - surface parser problems to the user
        raise ValidationError(f'Could not parse the file as CSV: {exc}') from exc
    if df.empty:
        raise ValidationError('The CSV has a header but no data rows.')
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean_rows(df):
    """Row-level checks. Returns (clean_df, rejected_reasons dict)."""
    bad = pd.Series(False, index=df.index)
    reasons = {}

    def flag(mask, why):
        nonlocal bad
        n = int((mask & ~bad).sum())
        if n:
            reasons[why] = reasons.get(why, 0) + n
        bad |= mask

    for col in NUMERIC_RAW:
        num = pd.to_numeric(df[col], errors='coerce')
        flag(num.isna() | (num < 0), f'{col}: not a non-negative number')
        df[col] = num
    flag(~df['age'].isin(AGE_MAP), "age: not a valid bracket such as '[60-70)'")
    for col in ('diabetesMed', 'change'):
        flag(df[col] == '', f'{col}: empty')
    return df[~bad].copy(), reasons


def validate_labeled(raw_bytes, existing_ids=frozenset()):
    """Validate a labelled retraining file. Returns (clean_df, report)."""
    df = _read_csv(raw_bytes)
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            f'Missing {len(missing)} required column(s): {", ".join(missing[:12])}'
            f'{" ..." if len(missing) > 12 else ""}. Use the same columns as the original '
            f'diabetic_data.csv (see the sample files).')
    received = len(df)
    unknown_labels = ~df['readmitted'].isin(VALID_LABELS)
    if unknown_labels.all():
        raise ValidationError("Column 'readmitted' must contain the labels '<30', '>30' or 'NO'.")
    reasons = {}
    if unknown_labels.any():
        reasons["readmitted: label not in {'<30','>30','NO'}"] = int(unknown_labels.sum())
    df = df[~unknown_labels].copy()
    df, row_reasons = _clean_rows(df)
    reasons.update(row_reasons)

    # Original exclusion rule: patients who died in hospital are not at risk of readmission.
    excl = df['discharge_disposition_id'].astype(int).isin(config.EXCLUDED_DISCHARGE_IDS)
    n_excl = int(excl.sum())
    df = df[~excl]

    # De-duplicate (within file and against rows already stored in the database).
    dup = 0
    if 'encounter_id' in df.columns:
        has_id = df['encounter_id'] != ''
        within = has_id & df.duplicated('encounter_id', keep='first')
        stored = has_id & df['encounter_id'].isin(existing_ids)
        dup = int((within | stored).sum())
        df = df[~(within | stored)]

    positives = int((df['readmitted'] == '<30').sum())
    negatives = len(df) - positives
    if len(df) < config.MIN_UPLOAD_ROWS:
        raise ValidationError(
            f'Only {len(df)} usable rows after validation; at least {config.MIN_UPLOAD_ROWS} are required.'
            + (f' Rejected: {reasons}' if reasons else ''))
    if positives < MIN_CLASS_ROWS or negatives < MIN_CLASS_ROWS:
        raise ValidationError(
            f"Need at least {MIN_CLASS_ROWS} rows of each class (readmitted '<30' vs others); "
            f'found {positives} positive and {negatives} negative.')
    for col in NUMERIC_RAW:
        df[col] = df[col].astype(int)
    report = {
        'rows_received': received, 'rows_stored': len(df),
        'rows_rejected': received - len(df) - dup - n_excl, 'rows_duplicate': dup,
        'rows_excluded_expired': n_excl, 'positives': positives, 'negatives': negatives,
        'reject_reasons': reasons,
        'ignored_columns': [c for c in df.columns if c not in RAW_COLUMNS],
    }
    return df, report


def validate_unlabeled(raw_bytes, max_rows=5000):
    """Validate a file for batch prediction (no label required)."""
    df = _read_csv(raw_bytes)
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f'Missing {len(missing)} required column(s): {", ".join(missing[:12])}'
                              f'{" ..." if len(missing) > 12 else ""}.')
    if len(df) > max_rows:
        raise ValidationError(f'Batch prediction is limited to {max_rows} rows per file.')
    df, reasons = _clean_rows(df)
    if df.empty:
        raise ValidationError(f'No valid rows found. Problems: {reasons}')
    return df, {'rows_valid': len(df), 'reject_reasons': reasons}
