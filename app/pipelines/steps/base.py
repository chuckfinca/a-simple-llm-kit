from typing import List
from app.core.pipeline import PipelineStep
from app.core.types import MediaType, PipelineData
import dspy

class BaseDSPyStep(PipelineStep):
    """Base class for pipeline steps that use DSPy modules."""
    def __init__(self, model_manager, model_id: str):
        self.model_manager = model_manager
        self.model_id = model_id
        
    @property
    def dspy_module(self):
        raise NotImplementedError
        
    async def process(self, data: PipelineData) -> PipelineData:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(self.dspy_module, lm)
            result = predictor(input=data.content)
            
            return PipelineData(
                media_type=self.output_media_type,
                content=result.output,
                metadata={**data.metadata, "model_used": self.model_id}
            )
    
    @property
    def output_media_type(self) -> MediaType:
        return MediaType.TEXT

class BaseImageStep(PipelineStep):
    """Base class for traditional image processing steps."""
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.IMAGE]