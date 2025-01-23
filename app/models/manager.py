import yaml
import logging
from contextlib import contextmanager
import dspy
from app.core.config import get_settings

class ModelManager:
    def __init__(self, config_path: str):
        self.settings = get_settings()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        api_keys = {
            'openai': self.settings.openai_api_key,
            'anthropic': self.settings.anthropic_api_key,
            'huggingface': self.settings.huggingface_api_key
        }
        
        for model_id, model_config in self.config['models'].items():
            try:
                provider = model_config['model_name'].split('/')[0]
                api_key = api_keys.get(provider)
                
                if not api_key:
                    raise ValueError(f"API key not found for provider: {provider}")
                
                lm = dspy.LM(
                    model_config['model_name'],
                    api_key=api_key,
                    max_tokens=model_config.get('max_tokens', 1000),
                    **model_config.get('additional_params', {})
                )
                self.models[model_id] = lm
                logging.info(f"Successfully initialized model: {model_id}")
            except Exception as e:
                logging.error(f"Failed to initialize model {model_id}: {str(e)}")
                raise

    @contextmanager
    def get_model(self, model_id: str):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        try:
            yield self.models[model_id]
        except Exception as e:
            logging.error(f"Error with model {model_id}: {str(e)}")
            raise