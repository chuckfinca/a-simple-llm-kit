import asyncio
import uuid
from collections.abc import Callable
from typing import Any

import dspy
from dspy.signatures.signature import Signature

from llm_server.core import logging
from llm_server.core.program_registry import ProgramRegistry
from llm_server.core.protocols import StorageAdapter
from llm_server.core.types import ProgramExecutionInfo, ProgramMetadata
from llm_server.core.utils import format_timestamp


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

    def _extract_model_info(self) -> dict[str, dict[str, str]]:
        model_info = {}
        try:
            config = self.model_manager.config
            for model_id, model_config in config.items():
                model_name = model_config.get("model_name", "")
                provider = model_name.split("/")[0] if "/" in model_name else "unknown"
                base_model = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )
                model_info[model_id] = {
                    "provider": provider,
                    "base_model": base_model,
                    "model_name": model_name,
                }
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

    async def execute_program(
        self,
        program_id: str,
        model_id: str,
        input_data: dict[str, Any],
        program_version: str = "latest",
        trace_id: str | None = None,
        preprocessor: Callable | None = None,
    ) -> tuple[Any, ProgramExecutionInfo, str | None]:
        """
        Executes a program and returns the result, execution info, and raw completion text.

        Note: The LM should be configured via dspy.context() before calling this method.
        """
        program_class = self.registry.get_program(program_id, program_version)
        if not program_class:
            raise ValueError(
                f"Program {program_id} version {program_version} not found"
            )

        if program_version == "latest":
            metadata = self.registry.get_program_metadata(program_id)
            if metadata:
                program_version = metadata.version

        program_metadata = self.registry.get_program_metadata(
            program_id, program_version
        )

        execution_info = ProgramExecutionInfo(
            program_id=program_id,
            program_version=program_version,
            program_name=program_metadata.name if program_metadata else program_id,
            model_id=model_id,
            model_info=self.model_info.get(model_id, {}),
            execution_id=str(uuid.uuid4()),
            timestamp=format_timestamp(),
            trace_id=trace_id,
        )

        # Apply the preprocessor if one was provided
        if preprocessor and "image" in input_data:
            input_data["image"] = preprocessor(input_data["image"])

        try:
            # Create predictor and execute using the LM from dspy.context()
            predictor = dspy.Predict(program_class)
            result = await asyncio.to_thread(predictor, **input_data)

            # Extract raw completion text from the current LM context
            raw_completion_text = None
            logging.info("Attempting to extract raw completion from LM history...")
            try:
                # Get the current LM from DSPy's context
                current_lm = dspy.settings.lm
                if hasattr(current_lm, "history") and current_lm.history:
                    last_interaction = current_lm.history[-1]

                    # Robustly check for the raw completion in multiple possible locations
                    if (
                        "response" in last_interaction
                        and "choices" in last_interaction["response"]
                        and last_interaction["response"]["choices"]
                    ):
                        raw_completion_text = last_interaction["response"]["choices"][
                            0
                        ].get("text")
                        logging.info(
                            "SUCCESS: Extracted raw completion from history['response']['choices']."
                        )
                    elif "outputs" in last_interaction and last_interaction["outputs"]:
                        raw_completion_text = last_interaction["outputs"][0]
                        logging.info(
                            "SUCCESS: Extracted raw completion from history['outputs']."
                        )
                    else:
                        logging.warning(
                            "History entry found, but a known key for raw output ('response' or 'outputs') is missing or empty."
                        )
                else:
                    logging.warning(
                        "LM history is missing or empty. Cannot extract raw completion."
                    )

            except Exception as e:
                logging.error(
                    f"ProgramManager failed to extract raw completion text: {e}",
                    exc_info=True,
                )

            self.executions.append(execution_info)
            return result, execution_info, raw_completion_text

        except Exception as e:
            logging.error(
                f"Error executing program {program_id}/{program_version}: {e}",
                exc_info=True,
            )
            raise

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
            {"model_id": model_id, **info} for model_id, info in self.model_info.items()
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
        self.registry.save_evaluation_result(
            program_id=program_id,
            version=program_version,
            model_id=model_id,
            model_info=self.model_info.get(model_id, {}),
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
