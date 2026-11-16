import joblib
import pandas as pd

from src.backend.config.settings import settings


# Load reduced deployment model package once
package = joblib.load(
    settings.deployment_model_path
)

model = package["model"]
imputer = package["imputer"]
features = package["features"]
threshold = package["threshold"]


def predict_mortality(input_data):

    # Convert input into one-row DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )

    # Keep exactly the trained 14 features
    input_df = input_df[features]

    # Apply same imputation used during training
    input_processed = imputer.transform(
        input_df
    )

    # Probability of Expired class
    probability = model.predict_proba(
        input_processed
    )[:, 1][0]

    # Apply final threshold
    if probability >= threshold:
        prediction = "Expired"
    else:
        prediction = "Alive"

    return prediction, probability, threshold