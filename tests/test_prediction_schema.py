from src.backend.schemas.prediction_schema import PredictionRequest

data = PredictionRequest(
    age=0,
    BUN_max=0,
    sao2_std=0,
    heartrate_mean=0,
    sao2_min=0,
    respiration_std=0,
    respiration_mean=0,
    RDW_min=0,
    heartrate_min=0,
    hospitaladmitoffset=0,
    heartrate_max=0,
    heartrate_std=0,
    calcium_mean=0,
    admissionweight=0
)

print(data)
print(data.model_dump())


