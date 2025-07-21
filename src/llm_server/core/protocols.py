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

    async def predict(self, input: Any) -> Any: ...

    def get_lm_history(self) -> list[Any]: ...
