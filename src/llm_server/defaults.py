from typing import Any

import yaml

from llm_server.core.protocols import ConfigProvider, StorageAdapter


class InMemoryStorageAdapter(StorageAdapter):
    """Simple in-memory storage for development/testing."""
    def __init__(self): self._data = {}
    def save(self, key: str, data: str) -> None: self._data[key] = data
    def load(self, key: str): return self._data.get(key)
    def list_keys(self, prefix: str = ""): return [k for k in self._data if k.startswith(prefix)]
    def delete(self, key: str): return self._data.pop(key, None) is not None


class YamlConfigProvider(ConfigProvider):
    """Loads model configuration from a standard YAML file."""
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    def get_models(self) -> dict[str, Any]: return self.config.get("models", {})
