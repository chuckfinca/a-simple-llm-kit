from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict
import os

class Settings(BaseSettings):
    config_path: str = "config/model_config.yaml"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")

    class Config:
        env_file = ".env"