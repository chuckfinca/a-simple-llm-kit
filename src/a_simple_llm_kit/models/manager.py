from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.config import FrameworkSettings
from a_simple_llm_kit.core.protocols import ConfigProvider
from a_simple_llm_kit.core.providers import ProviderManager


class ModelManager:
    def __init__(self, config_provider: ConfigProvider, settings: FrameworkSettings):
        self.settings = settings
        self.config_provider = config_provider
        self.config = self.config_provider.get_models()
        self.models = {}
        self.provider_manager = ProviderManager(self.settings)
        self._initialize_models()

    def _initialize_models(self):
        for model_id, model_config in self.config.items():
            try:
                lm = self.provider_manager.initialize_model(
                    model_config["model_name"], model_config
                )
                self.models[model_id] = lm
                logging.info(f"Successfully initialized model: {model_id}")
            except Exception as e:
                logging.error(f"Failed to initialize model {model_id}: {str(e)}")
                raise

    def get_model(self, model_id: str):
        """Get a model instance directly without context manager"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        return self.models[model_id]
