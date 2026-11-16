# This script keeps the final preprocessing workflow from the notebook:
# first-24-hour filtering, patient leakage removal, age conversion,
# vital/lab aggregation, merging, target mapping, high-cardinality removal,
# stratified split, completely-missing-column removal, preprocessing,
# and artifact saving.
#
# Intermediate print checks, duplicate checks, exploratory missing-value
# inspection, and repeated earlier preprocessing attempts are not included.
# This script contains the final reproducible preprocessing workflow.

# 1. Imports
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.training.config.settings import settings


# 2. Configuration
WINDOW = 1440


# 3. Load the three datasets
def load_data():
    patient = pd.read_csv(settings.patient_dataset_path)
    vital = pd.read_csv(settings.vital_dataset_path)
    lab = pd.read_csv(settings.lab_dataset_path)

    return patient, vital, lab


# 4. Preprocess patient data
def preprocess_patient(patient):
    target = settings.target_col
    patient_prep = patient.copy()

    # Remove rows where target is missing
    patient_prep = patient_prep[patient_prep[target].notna()].copy()

    # Leakage columns
    patient_leakage_cols = [
        "hospitaldischargestatus",
        "hospitaldischargelocation",
        "hospitaldischargeoffset",
        "unitdischargestatus",
        "unitdischargelocation",
        "unitdischargeoffset",
        "dischargeweight",
    ]

    patient_features = patient_prep.drop(columns=patient_leakage_cols, errors="ignore")
    patient_target = patient_prep[["patientunitstayid", target]].copy()

    # Convert age to numeric
    patient_features["age"] = pd.to_numeric(patient_features["age"], errors="coerce")

    return patient_features, patient_target


# 5. Preprocess vital-sign data
def preprocess_vitals(vital):
    vital_prep = vital.copy()

    # First 24 hours only
    vital_prep = vital_prep[
        (vital_prep["observationoffset"] >= 0)
        & (vital_prep["observationoffset"] <= WINDOW)
    ].copy()

    vital_feature_cols = [
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "cvp",
        "etco2",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
        "pasystolic",
        "padiastolic",
        "pamean",
        "st1",
        "st2",
        "st3",
        "icp",
    ]

    vital_agg = (
        vital_prep
        .groupby("patientunitstayid")[vital_feature_cols]
        .agg(["mean", "min", "max", "std"])
    )

    vital_agg.columns = [
        f"{column}_{stat}"
        for column, stat in vital_agg.columns
    ]

    vital_agg = vital_agg.reset_index()

    return vital_agg


# 6. Preprocess laboratory data
def preprocess_labs(lab):
    lab_prep = lab.copy()

    # First 24 hours only
    lab_prep = lab_prep[
        (lab_prep["labresultoffset"] >= 0)
        & (lab_prep["labresultoffset"] <= WINDOW)
    ].copy()

    # Remove missing lab results
    lab_prep = lab_prep[lab_prep["labresult"].notna()].copy()

    lab_agg = (
        lab_prep
        .groupby(["patientunitstayid", "labname"])["labresult"]
        .agg(["mean", "min", "max", "std"])
        .unstack()
    )

    lab_agg.columns = [
        f"{labname}_{stat}"
        for stat, labname in lab_agg.columns
    ]

    lab_agg = lab_agg.reset_index()

    return lab_agg


# 7. Merge patient, vital, and laboratory data
def create_final_dataset(patient_features, patient_target, vital_agg, lab_agg):
    vital_lab = pd.merge(vital_agg, lab_agg, on="patientunitstayid", how="outer")

    final_df = pd.merge(patient_features, vital_lab, on="patientunitstayid", how="inner")
    final_df = pd.merge(final_df, patient_target, on="patientunitstayid", how="inner")

    return final_df


# 8. Prepare features, target, and train-test split
def prepare_train_test(final_df):
    target = settings.target_col

    # Save merged raw dataset
    final_df.to_csv(settings.final_preprocessed_raw_path, index=False)

    # ID column has no clinical meaning
    final_df = final_df.drop(columns=["patientunitstayid"], errors="ignore")

    # Convert target to 0 / 1
    final_df[target] = final_df[target].map({
        "Alive": 0,
        "Expired": 1,
    })

    X = final_df.drop(columns=[target])
    y = final_df[target]

    # Remove high-cardinality/non-clinical columns
    high_cardinality_cols = [
        "hospitaladmittime24",
        "hospitaldischargetime24",
        "unitadmittime24",
        "unitdischargetime24",
        "uniquepid",
    ]

    X = X.drop(columns=high_cardinality_cols, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    # Remove columns completely missing in training data
    missing_all = X_train.isna().all()
    remove_missing_cols = X_train.columns[missing_all].tolist()

    X_train = X_train.drop(columns=remove_missing_cols)
    X_test = X_test.drop(columns=remove_missing_cols)

    return X_train, X_test, y_train, y_test


# 9. Build the preprocessing pipeline
def build_preprocessor(X_train):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            numerical_cols,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_cols,
        ),
    ])

    return preprocessor


# 10. Save preprocessing artifacts
def save_artifacts(preprocessor, X_train_processed, X_test_processed, X_test, y_train, y_test):
    joblib.dump(X_train_processed, settings.x_train_processed_path)
    joblib.dump(X_test_processed, settings.x_test_processed_path)
    joblib.dump(X_test, settings.x_test_raw_path)
    joblib.dump(y_train, settings.y_train_path)
    joblib.dump(y_test, settings.y_test_path)
    joblib.dump(preprocessor, settings.preprocessor_path)


# 11. Run the complete preprocessing workflow
def main():
    print("Loading datasets...")
    patient, vital, lab = load_data()

    print("Preprocessing patient data...")
    patient_features, patient_target = preprocess_patient(patient)

    print("Preprocessing vital data...")
    vital_agg = preprocess_vitals(vital)

    print("Preprocessing lab data...")
    lab_agg = preprocess_labs(lab)

    print("Creating final dataset...")
    final_df = create_final_dataset(patient_features, patient_target, vital_agg, lab_agg)
    print("Final merged shape:", final_df.shape)

    X_train, X_test, y_train, y_test = prepare_train_test(final_df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    preprocessor = build_preprocessor(X_train)

    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Processed X_train:", X_train_processed.shape)
    print("Processed X_test:", X_test_processed.shape)

    print("Saving preprocessing artifacts...")
    save_artifacts(preprocessor, X_train_processed, X_test_processed, X_test, y_train, y_test)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
