# This script is for the final model evaluation and final package saving.
#
# The notebook contains four separate activities:
# 1. Final evaluation at the selected threshold.
# 2. Feature importance and SHAP analysis.
# 3. Saving the final model package.
# 4. Testing a prediction on one patient.
#
# In this production training folder, this script contains only final model
# evaluation and final package saving. SHAP and explanation plots remain in
# the notebook.

# 1. Imports
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.training.config.settings import settings


# 2. Load trained model and test artifacts
def load_artifacts():
    model = joblib.load(settings.xgb_model_path)
    preprocessor = joblib.load(settings.preprocessor_path)
    X_test = joblib.load(settings.x_test_processed_path)
    y_test = joblib.load(settings.y_test_path)

    return model, preprocessor, X_test, y_test


# 3. Evaluate the model using the final threshold
def evaluate_model(model, X_test, y_test):
    # Get mortality probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Apply final threshold
    y_prediction = (y_probability >= settings.mortality_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction)
    recall = recall_score(y_test, y_prediction)
    f1 = f1_score(y_test, y_prediction)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n===== FINAL XGBOOST MODEL EVALUATION =====")
    print("Threshold:", settings.mortality_threshold)
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1 Score :", round(f1, 3))
    print("ROC-AUC  :", round(roc_auc, 3))

    print("\nClassification Report:")
    print(classification_report(y_test, y_prediction))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))


# 4. Save the final model package
def save_final_package(model, preprocessor):
    final_model_package = {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": settings.mortality_threshold,
    }

    joblib.dump(final_model_package, settings.final_model_path)

    print("\nFinal model package saved:", settings.final_model_path)


# 5. Run the evaluation and saving workflow
def main():
    print("Loading model and test artifacts...")
    model, preprocessor, X_test, y_test = load_artifacts()

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    evaluate_model(model, X_test, y_test)
    save_final_package(model, preprocessor)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
