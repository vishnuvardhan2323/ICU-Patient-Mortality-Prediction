import joblib

from src.backend.services.predictor import predict_mortality
from src.training.config.settings import settings


X_test_raw = joblib.load(
    settings.x_test_raw_path
)

test_patient = X_test_raw.iloc[0]

prediction, probability, threshold = predict_mortality(
    test_patient
)

print("Prediction:", prediction)

print(
    "Mortality Probability:",
    round(probability, 4)
)

print(
    "Threshold:",
    threshold
)