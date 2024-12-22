from typing import List
from app.core.pipeline import PipelineStep
from app.core.types import MediaType, PipelineData
from app.models.predictor import Predictor
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
            predictor = dspy.Predict(Predictor, lm)
            result = predictor(input=data.content)
            
            return PipelineData(
                media_type=MediaType.TEXT,
                content=result.output,
                metadata={
                    **data.metadata,
                    "model_used": self.model_id,
                }
            )

# Test code
async def test_predictor_pipeline():
    from app.core.pipeline import Pipeline
    from app.models.manager import ModelManager
    from app.core.config import Settings
    
    # Initialize model manager
    settings = Settings()
    model_manager = ModelManager(settings.config_path)
    
    # Create pipeline with predictor step
    pipeline = Pipeline([PredictorStep(model_manager, "gpt-4")])
    
    # Create initial data
    initial_data = PipelineData(
        media_type=MediaType.TEXT,
        content="Complete this: The quick brown fox...",
        metadata={"test": True}
    )
    
    # Execute pipeline
    result = await pipeline.execute(initial_data)
    print(f"Input: {initial_data.content}")
    print(f"Output: {result.content}")
    print(f"Metadata: {result.metadata}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_predictor_pipeline())