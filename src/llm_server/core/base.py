from llm_server.core.protocols import ModelBackend, PipelineStep
from llm_server.core.types import MediaType, PipelineData


class BaseModelStep(PipelineStep):
    """Base class for model-based pipeline steps"""

    def __init__(
        self,
        backend: ModelBackend,
        accepted_types: list[MediaType],
        output_type: MediaType,
    ):
        self._backend = backend
        self._accepted_types = accepted_types
        self._output_type = output_type

    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types

    async def process(self, data: PipelineData) -> PipelineData:
        result = await self._backend.predict(data.content)
        return PipelineData(
            media_type=self._output_type,
            content=result,
            metadata={**data.metadata, "processed": True},
        )
