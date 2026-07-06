from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/nextnews.db"

    azure_openai_llm_endpoint: str | None = None
    azure_openai_llm_api_key: SecretStr | None = None
    azure_openai_llm_deployment: str | None = None

    azure_openai_image_endpoint: str | None = None
    azure_openai_image_api_key: SecretStr | None = None
    azure_openai_image_deployment: str | None = None
    azure_openai_image_api_version: str = "2025-04-01-preview"

    hn_fetch_limit: int = Field(default=20, ge=1, le=500)
    hn_pipeline_interval_seconds: int = Field(default=1800, ge=60)

    image_output_dir: str = "./data/images"
    image_quality: Literal["low", "medium", "high", "auto"] = "medium"
    image_size: str = "1024x1024"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def azure_llm_configured(self) -> bool:
        return all(
            [
                self.azure_openai_llm_endpoint,
                self.azure_openai_llm_api_key,
                self.azure_openai_llm_deployment,
            ]
        )

    @property
    def azure_image_configured(self) -> bool:
        return all(
            [
                self.azure_openai_image_endpoint,
                self.azure_openai_image_api_key,
                self.azure_openai_image_deployment,
                self.azure_openai_image_api_version,
            ]
        )

    @property
    def azure_configured(self) -> bool:
        return self.azure_llm_configured and self.azure_image_configured

    @property
    def azure_openai_llm_api_key_value(self) -> str:
        if not self.azure_openai_llm_api_key:
            raise RuntimeError("AZURE_OPENAI_LLM_API_KEY is not configured")
        return self.azure_openai_llm_api_key.get_secret_value()

    @property
    def azure_openai_image_api_key_value(self) -> str:
        if not self.azure_openai_image_api_key:
            raise RuntimeError("AZURE_OPENAI_IMAGE_API_KEY is not configured")
        return self.azure_openai_image_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
