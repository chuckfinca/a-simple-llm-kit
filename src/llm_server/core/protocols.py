from typing import Any, Optional, Protocol, runtime_checkable

from llm_server.core.types import MediaType, PipelineData, ProgramMetadata


@runtime_checkable
class PipelineStep(Protocol):
    """Protocol defining what a pipeline step must implement"""

    async def process(self, data: PipelineData) -> PipelineData: ...

    @property
    def accepted_media_types(self) -> list[MediaType]: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model interaction implementations"""

    model_id: str

    program_metadata: Optional[ProgramMetadata]
    last_prompt_tokens: Optional[int]
    last_completion_tokens: Optional[int]

    async def predict(self, input: Any) -> Any: ...

    def get_lm_history(self) -> list[Any]: ...


class StorageAdapter(Protocol):
    """Defines the contract for how the framework stores and retrieves program metadata."""

    def save(self, key: str, data: str) -> None: ...
    def load(self, key: str) -> Optional[str]: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...
    def delete(self, key: str) -> bool: ...


class ConfigProvider(Protocol):
    """Defines the contract for how the framework gets model configurations."""

    def get_models(self) -> dict[str, Any]: ...


class OutputProcessor(Protocol):
    """Defines the contract for processing the raw output from a DSPy signature."""

    def process(self, result: Any) -> Any: ...
