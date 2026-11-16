# This file used for Backend for predictor file

# 1. Imports
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

from src.training.config.settings import settings


# 2. Final deployment features
FINAL_FEATURES = [
    "age",
    "BUN_max",
    "sao2_std",
    "heartrate_mean",
    "sao2_min",
    "respiration_std",
    "respiration_mean",
    "RDW_min",
    "heartrate_min",
    "hospitaladmitoffset",
    "heartrate_max",
    "heartrate_std",
    "calcium_mean",
    "admissionweight",
]


# 3. Load dataset
def load_data():
    df = pd.read_csv(settings.final_preprocessed_raw_path)
    return df


# 4. Prepare features and target, then split the data
def prepare_data(df):
    X = df[FINAL_FEATURES].copy()

    y = df[settings.target_col].map({
        "Alive": 0,
        "Expired": 1,
    })

    valid_rows = y.notna()

    X = X[valid_rows]
    y = y[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_state,
    )

    return X_train, X_test, y_train, y_test


# 5. Handle missing values using median imputation
def preprocess_data(X_train, X_test):
    imputer = SimpleImputer(strategy="median")

    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)

    return X_train_processed, X_test_processed, imputer


# 6. Build XGBoost model
def build_model():
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=1.0,
        scale_pos_weight=8,
        random_state=settings.random_state,
        eval_metric="logloss",
    )

    return model


# 7. Train the model
def train_model(model, X_train_processed, y_train):
    model.fit(X_train_processed, y_train)
    return model


# 8. Evaluate the model
def evaluate_model(model, X_test_processed, y_test):
    probability = model.predict_proba(X_test_processed)[:, 1]

    prediction = (probability >= settings.mortality_threshold).astype(int)

    accuracy = accuracy_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, probability)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)

    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(cm)


# 9. Save the complete deployment package
def save_model(model, imputer):
    deployment_package = {
        "model": model,
        "imputer": imputer,
        "features": FINAL_FEATURES,
        "threshold": settings.mortality_threshold,
    }

    joblib.dump(deployment_package, settings.deployment_model_path)

    print("Deployment model saved:")
    print(settings.deployment_model_path)


# 10. Run the complete training workflow
def main():
    print("Loading data...")
    df = load_data()

    print("Preparing 14 deployment features...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Applying median imputation...")
    X_train_processed, X_test_processed, imputer = preprocess_data(X_train, X_test)

    print("Building XGBoost model...")
    model = build_model()

    print("Training model...")
    model = train_model(model, X_train_processed, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test_processed, y_test)

    print("Saving deployment package...")
    save_model(model, imputer)


if __name__ == "__main__":
    main()
