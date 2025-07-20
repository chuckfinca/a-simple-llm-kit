from typing import Any, Protocol, Optional

from llm_server.core.types import MediaType, PipelineData, ProgramMetadata


class PipelineStep(Protocol):
    """Protocol defining what a pipeline step must implement"""

    async def process(self, data: PipelineData) -> PipelineData: ...

    @property
    def accepted_media_types(self) -> list[MediaType]: ...


class ModelBackend(Protocol):
    """Protocol for model interaction implementations"""

    program_metadata: Optional[ProgramMetadata]

    async def predict(self, input: Any) -> Any: ...
