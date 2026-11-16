from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Project
    project_root: str
    app_name: str
    environment: str

    # Logging
    log_path: str

    # Raw datasets
    patient_dataset_path: str
    vital_dataset_path: str
    lab_dataset_path: str

    # Final merged dataset
    final_preprocessed_raw_path: str

    # Models
    final_model_path: str
    preprocessor_path: str
    xgb_model_path: str

    # Training / Evaluation artifacts
    x_train_processed_path: str
    x_test_processed_path: str
    x_test_raw_path: str
    y_train_path: str
    y_test_path: str

    # ML configuration
    target_col: str
    mortality_threshold: float
    test_size: float
    random_state: int

    # API configuration
    api_host: str
    api_port: int
    api_prefix: str
    prediction_endpoint: str
    api_url: str

    deployment_model_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()