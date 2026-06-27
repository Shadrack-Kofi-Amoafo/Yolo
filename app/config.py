from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    roboflow_api_key: str
    model_id: str = "plant-disease-ep6jy/2"

    # serverless = Roboflow hosted API (easiest to start)
    # local      = self-hosted Roboflow Inference Server (e.g. Docker on :9001)
    inference_mode: str = "serverless"
    inference_api_url: str = "https://serverless.roboflow.com"

    confidence_threshold: float = 0.4
    host: str = "0.0.0.0"
    # Render and most PaaS platforms inject PORT at runtime.
    port: int = 8001

    @property
    def api_url(self) -> str:
        if self.inference_mode == "local":
            return self.inference_api_url
        return "https://serverless.roboflow.com"


settings = Settings()
