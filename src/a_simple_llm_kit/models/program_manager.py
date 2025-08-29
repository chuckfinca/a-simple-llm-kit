import uuid
from typing import Any

from dspy.signatures.signature import Signature

from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.program_registry import ProgramRegistry
from a_simple_llm_kit.core.protocols import StorageAdapter
from a_simple_llm_kit.core.types import ModelInfo, ProgramExecutionInfo, ProgramMetadata


class ProgramManager:
    """
    Manager for DSPy programs, handling registration, execution tracking, and versioning.
    """

    def __init__(self, model_manager, storage_adapter: StorageAdapter):
        """
        Initializes the ProgramManager.

        Args:
            model_manager: An instance of the ModelManager.
            storage_adapter: A concrete implementation of the StorageAdapter protocol
                             that defines how and where program metadata is stored.
        """
        self.model_manager = model_manager
        self.registry = ProgramRegistry(storage_adapter)
        self.executions: list[ProgramExecutionInfo] = []
        self.model_info = self._extract_model_info()

    def _extract_model_info(self) -> dict[str, ModelInfo]:
        model_info = {}
        try:
            config = self.model_manager.config
            for model_id, model_config in config.items():
                model_name = model_config.get("model_name", "")
                provider = model_name.split("/")[0] if "/" in model_name else "unknown"
                base_model = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )
                model_info[model_id] = ModelInfo(
                    provider=provider,
                    base_model=base_model,
                    model_name=model_name,
                )
        except Exception as e:
            logging.error(f"Error extracting model info from config: {e}")
        return model_info

    def register_program(
        self,
        program_class: type[Signature],
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> ProgramMetadata:
        return self.registry.register_program(
            program_class=program_class,
            name=name or program_class.__name__,
            description=description,
            tags=tags,
            version=version,
        )

    def get_program(
        self, program_id: str, version: str = "latest"
    ) -> type[Signature] | None:
        return self.registry.get_program(program_id, version)

    def get_execution_history(
        self,
        program_id: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[ProgramExecutionInfo]:
        filtered = self.executions
        if program_id:
            filtered = [e for e in filtered if e.program_id == program_id]
        if model_id:
            filtered = [e for e in filtered if e.model_id == model_id]
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]

    def register_optimized_program(
        self,
        program_class: type[Signature],
        parent_id: str,
        optimizer_name: str,
        parent_version: str = "latest",
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> ProgramMetadata:
        if parent_version == "latest":
            parent_metadata = self.registry.get_program_metadata(parent_id)
            if not parent_metadata:
                raise ValueError(f"Parent program {parent_id} not found")
            parent_version = parent_metadata.version
        return self.registry.register_optimized_program(
            program_class=program_class,
            parent_id=parent_id,
            parent_version=parent_version,
            optimizer_name=optimizer_name,
            name=name,
            description=description,
            tags=tags,
        )

    def get_program_tree(self, program_id: str) -> dict[str, Any]:
        return self.registry.get_program_tree(program_id)

    def get_available_models(self) -> list[dict[str, Any]]:
        return [
            {"model_id": model_id, **info.model_dump(by_alias=True)}
            for model_id, info in self.model_info.items()
        ]

    def save_evaluation_result(
        self,
        program_id: str,
        model_id: str,
        results: dict[str, Any],
        program_version: str = "latest",
        evaluation_id: str | None = None,
    ):
        if program_version == "latest":
            program_metadata = self.registry.get_program_metadata(program_id)
            if not program_metadata:
                raise ValueError(f"Program {program_id} not found")
            program_version = program_metadata.version

        model_info_obj = self.model_info.get(model_id)
        model_info_dict = (
            model_info_obj.model_dump(by_alias=True) if model_info_obj else {}
        )

        self.registry.save_evaluation_result(
            program_id=program_id,
            version=program_version,
            model_id=model_id,
            model_info=model_info_dict,
            evaluation_id=evaluation_id or str(uuid.uuid4()),
            results=results,
        )

    def get_evaluation_results(
        self,
        program_id: str,
        version: str | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.registry.get_evaluation_results(
            program_id=program_id, version=version, model_id=model_id
        )
