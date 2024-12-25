from typing import Protocol, Any, TypeVar, Dict
from app.core.types import MediaType, PipelineData

T = TypeVar('T')
U = TypeVar('U')

class Processor(Protocol):
    """Protocol defining what a processor must implement"""
    async def process(self, data: PipelineData) -> PipelineData:
        ...
    
    @property
    def accepted_media_types(self) -> list[MediaType]:
        ...

class ModelBackend(Protocol):
    """Protocol for model interaction implementations"""
    async def predict(self, input: Any) -> Any:
        ...