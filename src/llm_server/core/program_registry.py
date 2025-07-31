# llm-server/src/llm_server/core/program_registry.py

import hashlib
import importlib
import inspect
import json
from typing import Optional

from llm_server.core import logging
from llm_server.core.model_interfaces import Signature
from llm_server.core.protocols import StorageAdapter  # <-- Uses the new protocol
from llm_server.core.types import ProgramMetadata
from llm_server.core.utils import format_timestamp


class ProgramRegistry:
    """Registry for managing DSPy program signatures and their versions."""

    def __init__(self, storage_adapter: StorageAdapter):
        """
        Initializes the registry with a storage adapter for persistence.
        
        Args:
            storage_adapter: An object that conforms to the StorageAdapter protocol.
        """
        self.storage_adapter = storage_adapter
        self.programs: dict[str, dict[str, type[Signature]]] = {}
        self._load_programs()

    def _load_programs(self):
        """Load existing programs from the provided storage adapter."""
        # The key format is expected to be "program_id/version.json"
        all_keys = self.storage_adapter.list_keys()
        version_keys = [k for k in all_keys if k.endswith('.json') and '/' in k]

        for key in version_keys:
            try:
                program_id, version_str = key.rsplit('/', 1)
                version = version_str.replace('.json', '')
                
                raw_data = self.storage_adapter.load(key)
                if not raw_data:
                    logging.warning(f"Could not load data for key: {key}")
                    continue

                program_data = json.loads(raw_data)
                
                module_path = program_data.get("module_path")
                class_name = program_data.get("class_name")

                if module_path and class_name:
                    try:
                        module = importlib.import_module(module_path)
                        program_class = getattr(module, class_name)
                        
                        if program_id not in self.programs:
                            self.programs[program_id] = {}
                        self.programs[program_id][version] = program_class
                    except (ImportError, AttributeError) as e:
                        logging.warning(f"Failed to dynamically load program {program_id}/{version}: {e}")
            except Exception as e:
                logging.error(f"Error processing program from storage key '{key}': {e}")

    def register_program(
        self,
        program_class: type[Signature],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        version: str = "1.0.0",
        parent_id: Optional[str] = None,
        parent_version: Optional[str] = None,
    ) -> ProgramMetadata:
        """
        Register a DSPy program signature class and persist its metadata.
        """
        name = name or program_class.__name__
        program_id = self._generate_program_id(name)
        source_code = inspect.getsource(program_class)
        code_hash = hashlib.sha256(source_code.encode()).hexdigest()[:8]

        metadata = {
            "id": program_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "version": version,
            "code_hash": code_hash,
            "parent_id": parent_id,
            "parent_version": parent_version,
            "class_name": program_class.__name__,
            "module_path": program_class.__module__,
            "created_at": format_timestamp(),
            "source_code": source_code,
        }

        # Use the storage adapter to save the metadata
        storage_key = f"{program_id}/{version}.json"
        self.storage_adapter.save(storage_key, json.dumps(metadata, indent=2))

        # Add to in-memory registry
        if program_id not in self.programs:
            self.programs[program_id] = {}
        self.programs[program_id][version] = program_class

        return ProgramMetadata(**{k: v for k, v in metadata.items() if k in ProgramMetadata.model_fields})


    def get_program(
        self, program_id: str, version: str = "latest"
    ) -> Optional[type[Signature]]:
        """
        Get a program by ID and version from the in-memory cache.
        """
        if program_id not in self.programs:
            return None

        if version == "latest":
            versions = sorted(
                self.programs[program_id].keys(),
                key=lambda v: [int(x) for x in v.split(".")],
            )
            if not versions:
                return None
            version = versions[-1]

        return self.programs[program_id].get(version)


    def get_program_metadata(
        self, program_id: str, version: str = "latest"
    ) -> Optional[ProgramMetadata]:
        """
        Get program metadata by ID and version by loading from storage.
        """
        if program_id not in self.programs:
            return None

        if version == "latest":
            versions = sorted(
                self.programs[program_id].keys(),
                key=lambda v: [int(x) for x in v.split(".")],
            )
            if not versions:
                return None
            version = versions[-1]

        storage_key = f"{program_id}/{version}.json"
        raw_data = self.storage_adapter.load(storage_key)
        
        if not raw_data:
            return None

        try:
            data = json.loads(raw_data)
            # Filter to only include fields defined in the Pydantic model
            valid_fields = {k: v for k, v in data.items() if k in ProgramMetadata.model_fields}
            return ProgramMetadata(**valid_fields)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Error loading program metadata for {program_id}/{version}: {e}")
            return None


    def list_programs(self, tags: Optional[list[str]] = None) -> list[ProgramMetadata]:
        """
        List all registered programs, optionally filtered by tags.
        """
        result = []
        for program_id in self.programs:
            metadata = self.get_program_metadata(program_id) # Gets the latest version
            if metadata:
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue
                result.append(metadata)
        return result


    def _generate_program_id(self, name: str) -> str:
        """Generate a unique program ID based on the name."""
        base_id = "".join(
            c if c.isalnum() or c == "_" else "" for c in name.replace(" ", "_")
        ).lower()
        
        # In this model, we don't check for existence because the application
        # layer is responsible for ensuring names are unique if desired.
        return base_id

    # You can add back other helper methods like register_optimized_program,
    # list_program_versions, etc., following the same pattern of using
    # self.storage_adapter for all persistence.
