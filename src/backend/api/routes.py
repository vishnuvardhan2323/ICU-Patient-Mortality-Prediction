from fastapi import APIRouter

from src.backend.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
)

from src.backend.services.predictor import predict_mortality


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    input_data = request.model_dump()

    prediction, probability, threshold = predict_mortality(
        input_data
    )

    return PredictionResponse(
        prediction=prediction,
        mortality_probability=float(probability),
        threshold=float(threshold)
    )