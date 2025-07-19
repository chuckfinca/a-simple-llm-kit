from typing import Protocol, Any, List
from llm_server.core.types import MediaType, PipelineData

class PipelineStep(Protocol):
    """Protocol defining what a pipeline step must implement"""
    async def process(self, data: PipelineData) -> PipelineData:
        ...
    
    @property
    def accepted_media_types(self) -> List[MediaType]:
        ...

class ModelBackend(Protocol):
    """Protocol for model interaction implementations"""
    async def predict(self, input: Any) -> Any:
        ...