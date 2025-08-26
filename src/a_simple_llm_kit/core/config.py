import os

from pydantic_settings import BaseSettings


class FrameworkSettings(BaseSettings):
    config_path: str = "config/model_config.yml"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # --- OpenTelemetry Configuration ---
    # Master switch for the entire OTel integration
    otel_enabled: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"

    # Service identifiers used for resource attributes
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "a-simple-llm-kit-app")
    otel_service_version: str = os.getenv("OTEL_SERVICE_VERSION", "0.2.0")

    class ConfigDict:
        env_file = ".env"
        extra = "ignore"
