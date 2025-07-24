import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    config_path: str = "config/model_config.yml"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    llm_server_api_key: str = os.getenv("LLM_SERVER_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    log_level: str = "INFO"
    log_request_bodies: bool = False

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
