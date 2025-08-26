import hashlib
import importlib
import inspect
import json
from typing import Any

from dspy.signatures.signature import Signature

from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.protocols import StorageAdapter
from a_simple_llm_kit.core.types import ProgramMetadata
from a_simple_llm_kit.core.utils import format_timestamp


class ProgramRegistry:
    """Registry for managing DSPy program signatures and their versions."""

    def __init__(self, storage_adapter: StorageAdapter):
        """
        Initializes the registry with a storage adapter for persistence.
        """
        self.storage_adapter = storage_adapter
        self.programs: dict[str, dict[str, type[Signature]]] = {}
        self._load_programs()

    def _load_programs(self):
        """Load existing programs from the provided storage adapter."""
        metadata_keys = [
            k
            for k in self.storage_adapter.list_keys()
            if "/" in k and k.endswith(".json") and not k.startswith("evaluations/")
        ]

        for key in metadata_keys:
            try:
                raw_data = self.storage_adapter.load(key)
                if not raw_data:
                    continue

                program_data = json.loads(raw_data)
                program_id = program_data.get("id")
                version = program_data.get("version")
                module_path = program_data.get("module_path")
                class_name = program_data.get("class_name")

                if all([program_id, version, module_path, class_name]):
                    try:
                        module = importlib.import_module(module_path)
                        program_class = getattr(module, class_name)
                        if program_id not in self.programs:
                            self.programs[program_id] = {}
                        self.programs[program_id][version] = program_class
                    except (ImportError, AttributeError) as e:
                        logging.warning(
                            f"Failed to dynamically load program {program_id}/{version}: {e}"
                        )
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Error processing program from storage key '{key}': {e}")

    def register_program(
        self,
        program_class: type[Signature],
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        version: str = "1.0.0",
        parent_id: str | None = None,
        parent_version: str | None = None,
    ) -> ProgramMetadata:
        """Register a DSPy program signature class and persist its metadata."""
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

        storage_key = f"{program_id}/{version}.json"
        self.storage_adapter.save(storage_key, json.dumps(metadata, indent=2))

        if program_id not in self.programs:
            self.programs[program_id] = {}
        self.programs[program_id][version] = program_class
        return ProgramMetadata(
            **{k: v for k, v in metadata.items() if k in ProgramMetadata.model_fields}
        )

    def get_program(
        self, program_id: str, version: str = "latest"
    ) -> type[Signature] | None:
        """Get a program class by ID and version."""
        if program_id not in self.programs:
            logging.warning(
                f"Program ID '{program_id}' not found in in-memory registry."
            )
            return None

        version_to_get = version
        if version_to_get == "latest":
            # Resolve the latest version string first
            available_versions = self.list_program_versions(program_id)
            if not available_versions:
                logging.warning(f"No versions found for program ID '{program_id}'.")
                return None
            version_to_get = available_versions[-1]

        # Now, version_to_get is guaranteed to be a string.
        # We can safely use it to access the dictionary.
        program_class = self.programs[program_id].get(version_to_get)

        if program_class is None:
            logging.warning(
                f"Version '{version_to_get}' for program '{program_id}' not found in registry."
            )

        return program_class

    def get_program_metadata(
        self, program_id: str, version: str = "latest"
    ) -> ProgramMetadata | None:
        """Get program metadata by loading from storage."""
        if program_id not in self.programs and version != "latest":
            # If we don't have it in memory, we can't look it up by version
            return None
        if version == "latest":
            versions = self.list_program_versions(program_id)
            if not versions:
                return None
            version = versions[-1]

        storage_key = f"{program_id}/{version}.json"
        raw_data = self.storage_adapter.load(storage_key)
        if not raw_data:
            return None
        try:
            data = json.loads(raw_data)
            return ProgramMetadata(
                **{k: v for k, v in data.items() if k in ProgramMetadata.model_fields}
            )
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(
                f"Error loading program metadata for {program_id}/{version}: {e}"
            )
            return None

    def register_optimized_program(
        self,
        program_class: type[Signature],
        parent_id: str,
        parent_version: str,
        optimizer_name: str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> ProgramMetadata:
        """Registers an optimized version of a program."""
        parent_metadata = self.get_program_metadata(parent_id, parent_version)
        if not parent_metadata:
            raise ValueError(f"Parent program {parent_id} not found")

        major, minor, _ = map(int, parent_metadata.version.split("."))
        new_version = f"{major}.{minor + 1}.0"

        if name is None:
            name = f"{parent_metadata.name}_optimized"
        if tags is None:
            tags = []
        tags.append(f"optimizer:{optimizer_name}")

        return self.register_program(
            program_class=program_class,
            name=name,
            description=description,
            tags=tags,
            version=new_version,
            parent_id=parent_id,
            parent_version=parent_version,
        )

    def get_program_tree(self, program_id: str) -> dict[str, Any]:
        """Gets a hierarchical view of a program and its optimized children."""
        root_metadata = self.get_program_metadata(program_id)
        if not root_metadata:
            return {}

        tree = {
            "metadata": root_metadata.model_dump(),
            "versions": {},
            "derivatives": [],
        }
        all_versions = self.list_program_versions(program_id, with_metadata=True)
        for metadata in all_versions:
            tree["versions"][metadata.version] = metadata.model_dump()
        # A full implementation would recursively search for derivatives. This is a simplified version.
        return tree

    def save_evaluation_result(
        self,
        program_id: str,
        version: str,
        model_id: str,
        model_info: dict,
        evaluation_id: str,
        results: dict[str, Any],
    ):
        """Saves the result of a program evaluation to storage."""
        eval_data = {
            "evaluation_id": evaluation_id,
            "program_id": program_id,
            "version": version,
            "model_id": model_id,
            "model_info": model_info,
            "results": results,
            "evaluated_at": format_timestamp(),
        }
        storage_key = (
            f"evaluations/{program_id}/{version}/{model_id}/{evaluation_id}.json"
        )
        self.storage_adapter.save(storage_key, json.dumps(eval_data, indent=2))
        logging.info(f"Saved evaluation result to {storage_key}")

    def get_evaluation_results(
        self,
        program_id: str,
        version: str | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves evaluation results from storage."""
        prefix = f"evaluations/{program_id}/"
        if version:
            prefix += f"{version}/"
        if model_id:
            prefix += f"{model_id}/"

        eval_keys = [
            key
            for key in self.storage_adapter.list_keys(prefix=prefix)
            if key.endswith(".json")
        ]
        results = []
        for key in eval_keys:
            raw_data = self.storage_adapter.load(key)
            if raw_data:
                try:
                    results.append(json.loads(raw_data))
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse evaluation file: {key}")
        return results

    def list_program_versions(
        self, program_id: str, with_metadata: bool = False
    ) -> list:
        """Helper to list all version strings or metadata for a given program ID."""
        prefix = f"{program_id}/"
        all_keys = self.storage_adapter.list_keys(prefix=prefix)
        version_keys = [k for k in all_keys if k.endswith(".json")]
        versions = [
            key.replace(prefix, "").replace(".json", "") for key in version_keys
        ]

        sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split(".")])

        if not with_metadata:
            return sorted_versions

        return [
            self.get_program_metadata(program_id, v)
            for v in sorted_versions
            if self.get_program_metadata(program_id, v)
        ]

    def _generate_program_id(self, name: str) -> str:
        """Generate a unique program ID based on the name."""
        return "".join(
            c for c in name.replace(" ", "_") if c.isalnum() or c == "_"
        ).lower()
