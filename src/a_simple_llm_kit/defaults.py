from typing import Any

import yaml

from a_simple_llm_kit.core.protocols import ConfigProvider


class YamlConfigProvider(ConfigProvider):
    """Loads model configuration from a standard YAML file."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get_models(self) -> dict[str, Any]:
        return self.config.get("models", {})
