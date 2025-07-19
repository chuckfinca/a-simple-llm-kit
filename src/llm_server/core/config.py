from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict
import os

class Settings(BaseSettings):
    config_path: str = "config/model_config.yml"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    llm_server_api_key: str = os.getenv("LLM_SERVER_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()