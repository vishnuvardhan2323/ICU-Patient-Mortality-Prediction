from pydantic_settings import BaseSettings
class Settings(BaseSettings):

    log_path: str

    final_model_path: str

    deployment_model_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()