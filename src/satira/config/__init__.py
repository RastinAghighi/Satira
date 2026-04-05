from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql://satira:satira@localhost:5432/satira"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    s3_bucket: str = "satira-artifacts"

    # ML tracking
    mlflow_tracking_uri: str = "http://localhost:5000"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True


settings = Settings()
