"""Feature engineering from the original Intro-to-ML notebook (unchanged logic).

``engineer_features`` and ``parse_icd9_code`` are the same functions used to
train the original model, so training, retraining and serving all share one
preprocessing definition.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AGE_MAP = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
           '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
           '[80-90)': 85, '[90-100)': 95}
TOP_SPECIALTIES = ['InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
                   'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
                   'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology']

MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone']

# Columns of the raw UCI file (all of them, in order).
RAW_COLUMNS = (
    ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
     'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
     'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures',
     'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
     'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
     'max_glu_serum', 'A1Cresult'] + MEDICATION_COLUMNS +
    ['change', 'diabetesMed', 'readmitted'])

# Columns an uploaded labeled file MUST contain (what the model actually uses
# + the label). Identifier columns are optional; patient_nbr is used for a
# patient-grouped validation split when present.
MODEL_INPUT_COLUMNS = [c for c in RAW_COLUMNS
                       if c not in ('encounter_id', 'patient_nbr', 'weight',
                                    'payer_code', 'readmitted')]
REQUIRED_UPLOAD_COLUMNS = MODEL_INPUT_COLUMNS + ['readmitted']

NUMERIC_RAW = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
               'time_in_hospital', 'num_lab_procedures', 'num_procedures',
               'num_medications', 'number_outpatient', 'number_emergency',
               'number_inpatient', 'number_diagnoses']

# Neutral defaults used by the single-patient form for fields not shown.
RAW_DEFAULTS = {
    'race': 'Caucasian', 'gender': 'Female', 'age': '[60-70)',
    'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7,
    'time_in_hospital': 3, 'medical_specialty': '?', 'num_lab_procedures': 40,
    'num_procedures': 0, 'num_medications': 15, 'number_outpatient': 0,
    'number_emergency': 0, 'number_inpatient': 0, 'diag_1': '250', 'diag_2': '?',
    'diag_3': '?', 'number_diagnoses': 7, 'max_glu_serum': 'None',
    'A1Cresult': 'None', 'change': 'No', 'diabetesMed': 'Yes',
    **{m: 'No' for m in MEDICATION_COLUMNS},
}


def parse_icd9_code(code):
    """Map an ICD-9 code to a broad diagnostic group (Strack et al., 2014)."""
    if pd.isna(code) or code == '?' or code == '' or code is None:
        return 'Missing'
    code_str = str(code).strip()
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
    try:
        val = float(code_str)
    except ValueError:
        return 'Other'
    if (390 <= val <= 459) or val == 785:
        return 'Circulatory'
    elif (460 <= val <= 519) or val == 786:
        return 'Respiratory'
    elif (520 <= val <= 579) or val == 787:
        return 'Digestive'
    elif int(val) == 250:
        return 'Diabetes'
    elif 800 <= val <= 999:
        return 'Injury'
    elif 710 <= val <= 739:
        return 'Musculoskeletal'
    elif (580 <= val <= 629) or val == 788:
        return 'Genitourinary'
    elif 140 <= val <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def engineer_features(df):
    """Clinical feature engineering (identical to the original notebook)."""
    df = df.copy()
    df['age_num'] = df['age'].map(AGE_MAP).fillna(65).astype(float)
    df['diag_1_group'] = df['diag_1'].apply(parse_icd9_code)
    df['diag_2_group'] = df['diag_2'].apply(parse_icd9_code)
    df['diag_3_group'] = df['diag_3'].apply(parse_icd9_code)

    df['total_visits_prior'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    df['medication_density'] = df['num_medications'] / np.maximum(df['time_in_hospital'], 1)
    df['procedure_to_med_ratio'] = (df['num_procedures'] + df['num_lab_procedures']) / np.maximum(df['num_medications'], 1)
    df['med_change_flag'] = (df['change'] == 'Ch').astype(int)
    df['diabetes_med_flag'] = (df['diabetesMed'] == 'Yes').astype(int)

    df['medical_specialty_clean'] = df['medical_specialty'].apply(
        lambda x: 'Missing' if x == '?' or pd.isna(x) else (x if x in TOP_SPECIALTIES else 'Other'))
    df['admission_type_str'] = 'AdmType_' + df['admission_type_id'].astype(str)
    df['discharge_disp_str'] = 'DischDisp_' + df['discharge_disposition_id'].astype(str)
    df['admission_source_str'] = 'AdmSource_' + df['admission_source_id'].astype(str)

    drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'examide', 'citoglipton',
                 'readmitted', 'target', 'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
                 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def build_preprocessor(X_train_raw):
    """ColumnTransformer exactly as in the notebook (fit on training data only)."""
    numeric_cols = X_train_raw.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         categorical_cols)
    ], remainder='drop')


def complete_record(record):
    """Fill missing raw fields with neutral defaults and coerce numeric types."""
    row = dict(RAW_DEFAULTS)
    row.update({k: v for k, v in record.items() if v is not None and v != ''})
    for col in NUMERIC_RAW:
        row[col] = int(float(row[col]))
    return row
