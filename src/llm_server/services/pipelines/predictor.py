from typing import List
from llm_server.core.pipeline import PipelineStep
from llm_server.core.types import MediaType, PipelineData
from llm_server.models.predictor import Predictor
import dspy

class PredictorStep(PipelineStep):
    """Pipeline step that uses DSPy Predictor for text completion."""
    def __init__(self, model_manager, model_id: str):
        self.model_manager = model_manager
        self.model_id = model_id

    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.TEXT]

    async def process(self, data: PipelineData) -> PipelineData:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(Predictor)
            # Remove await since dspy.Predict is synchronous
            result = predictor(input=data.content)
            
            return PipelineData(
                media_type=MediaType.TEXT,
                content=result.output,
                metadata={
                    **data.metadata,
                    "model_used": self.model_id,
                }
            )