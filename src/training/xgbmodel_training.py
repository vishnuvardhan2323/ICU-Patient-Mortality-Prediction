# This file is used only for testing the XGBoost model, not for deployment.
#
# Logistic Regression, Random Forest, RandomizedSearchCV, threshold analysis,
# ROC plots, and confusion-matrix experiments remain in the model-training
# notebook. This script trains the already-selected XGBoost model only.

# 1. Imports
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.training.config.settings import settings


# 2. Load processed training and testing data
def load_training_data():
    X_train = joblib.load(settings.x_train_processed_path)
    X_test = joblib.load(settings.x_test_processed_path)
    y_train = joblib.load(settings.y_train_path)
    y_test = joblib.load(settings.y_test_path)

    return X_train, X_test, y_train, y_test


# 3. Build XGBoost model
def build_model():
    model = XGBClassifier(
        subsample=0.9,
        scale_pos_weight=8,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        colsample_bytree=1.0,
        random_state=settings.random_state,
        eval_metric="logloss",
    )

    return model


# 4. Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# 5. Evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    ypred_train = model.predict(X_train)
    ypred_test = model.predict(X_test)
    yprob_test = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, ypred_train)
    test_accuracy = accuracy_score(y_test, ypred_test)
    roc_auc = roc_auc_score(y_test, yprob_test)

    print("Train Accuracy:", round(train_accuracy, 4))
    print("Test Accuracy:", round(test_accuracy, 4))
    print("Test ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, ypred_test))

    return yprob_test


# 6. Save the standalone XGBoost model
def save_model(model):
    joblib.dump(model, settings.xgb_model_path)
    print("Model saved:", settings.xgb_model_path)


# 7. Run the XGBoost testing workflow
def main():
    print("Loading processed training data...")
    X_train, X_test, y_train, y_test = load_training_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Building XGBoost model...")
    model = build_model()

    print("Training model...")
    model = train_model(model, X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_train, X_test, y_train, y_test)

    print("Saving model...")
    save_model(model)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
