from pathlib import Path

from a_simple_llm_kit.core.protocols import StorageAdapter


class InMemoryStorageAdapter(StorageAdapter):
    """Simple in-memory storage for development and testing."""

    def __init__(self):
        self._data: dict[str, str] = {}

    def save(self, key: str, data: str) -> None:
        self._data[key] = data

    def load(self, key: str) -> str | None:
        return self._data.get(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._data if k.startswith(prefix)]

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None


class FileSystemStorageAdapter(StorageAdapter):
    """Stores program metadata on the local filesystem."""

    def __init__(self, base_dir: str = "dspy_programs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        return self.base_dir / key

    def save(self, key: str, data: str) -> None:
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")

    def load(self, key: str) -> str | None:
        path = self._get_path(key)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def list_keys(self, prefix: str = "") -> list[str]:
        prefix_path = self.base_dir / prefix
        if not prefix_path.is_dir():
            return []

        keys = []
        for path in prefix_path.rglob("*"):
            if path.is_file():
                # Store keys with relative paths and forward slashes
                keys.append(path.relative_to(self.base_dir).as_posix())

        # Also include files directly under the prefix if it's not the root
        if prefix:
            for path in self.base_dir.glob(f"{prefix}*"):
                if path.is_file():
                    key = path.relative_to(self.base_dir).as_posix()
                    if key not in keys:
                        keys.append(key)

        return keys

    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
