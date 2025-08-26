from dataclasses import dataclass, field
from typing import Any

import dspy

from a_simple_llm_kit.core.config import FrameworkSettings


@dataclass
class ProviderConfig:
    """Configuration for a model provider"""

    api_key: str
    base_url: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {}


class ProviderManager:
    """Manages provider configurations and initialization"""

    def __init__(self, settings: FrameworkSettings):
        self.providers = {
            "openai": ProviderConfig(
                api_key=settings.openai_api_key, default_params={}
            ),
            "anthropic": ProviderConfig(
                api_key=settings.anthropic_api_key, default_params={}
            ),
            "huggingface": ProviderConfig(
                api_key=settings.huggingface_api_key, default_params={}
            ),
            "gemini": ProviderConfig(
                api_key=settings.gemini_api_key, default_params={}
            ),
        }

    def get_provider_config(self, model_name: str) -> ProviderConfig:
        """Get provider configuration based on model name"""
        provider = model_name.split("/")[0]
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        return self.providers[provider]

    def initialize_model(
        self, model_name: str, model_config: dict[str, Any]
    ) -> dspy.LM:
        """Initialize a model with provider-specific configuration"""
        provider_config = self.get_provider_config(model_name)

        # Start with provider default params
        params = provider_config.default_params.copy()

        # Update with model-specific params, letting them override provider defaults
        model_params = model_config.get("additional_params", {})
        params.update(model_params)

        return dspy.LM(
            model_name,
            api_key=provider_config.api_key,
            max_tokens=model_config.get("max_tokens", 1000),
            **params,
        )
