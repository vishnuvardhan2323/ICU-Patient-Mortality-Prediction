from pydantic import BaseModel


class PredictionRequest(BaseModel):

    age: float
    BUN_max: float
    sao2_std: float
    heartrate_mean: float
    sao2_min: float
    respiration_std: float
    respiration_mean: float
    RDW_min: float
    heartrate_min: float
    hospitaladmitoffset: float
    heartrate_max: float
    heartrate_std: float
    calcium_mean: float
    admissionweight: float


class PredictionResponse(BaseModel):

    prediction: str
    mortality_probability: float
    threshold: float