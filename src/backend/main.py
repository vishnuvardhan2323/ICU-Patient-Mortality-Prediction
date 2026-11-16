from fastapi import FastAPI

from src.backend.api.routes import router


app = FastAPI(
    title="ICU Patient Mortality Risk Prediction"
)

app.include_router(
    router,
    prefix="/api"
)