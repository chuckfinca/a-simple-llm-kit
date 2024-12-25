from typing import Any
import dspy
from PIL import Image
import io

from app.core.protocols import ModelBackend
from app.core.types import MediaType, PipelineData

class DSPyBackend:
    """DSPy-based model implementation"""
    def __init__(self, model_manager, model_id: str, module_class):
        self.model_manager = model_manager
        self.model_id = model_id
        self.module_class = module_class
    
    async def predict(self, input: Any) -> Any:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(self.module_class, lm)
            result = predictor(input=input)
            return result.output

class ModelProcessor:
    """Standard processor for model-based operations"""
    def __init__(self, backend: ModelBackend, accepted_types: list[MediaType], output_type: MediaType):
        self.backend = backend
        self._accepted_types = accepted_types
        self.output_type = output_type
    
    async def process(self, data: PipelineData) -> PipelineData:
        result = await self.backend.predict(data.content)
        return PipelineData(
            media_type=self.output_type,
            content=result,
            metadata={**data.metadata, "processed": True}
        )
    
    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types

class ImagePreprocessor:
    """Image preprocessing operations"""
    def __init__(self, target_size: tuple[int, int] = (800, 800)):
        self.target_size = target_size
    
    async def process(self, data: PipelineData) -> PipelineData:
        image = Image.open(io.BytesIO(data.content))
        processed = image.resize(self.target_size)
        buffer = io.BytesIO()
        processed.save(buffer, format="PNG")
        
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=buffer.getvalue(),
            metadata={**data.metadata, "preprocessed": True}
        )
    
    @property
    def accepted_media_types(self) -> list[MediaType]:
        return [MediaType.IMAGE]
