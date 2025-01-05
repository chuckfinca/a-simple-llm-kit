from app.core.implementations import DSPyBackend, ImageConverterStep, ImageTypeValidator, ModelProcessor
from app.core.pipeline import Pipeline
from app.core.protocols import Processor
from app.core.types import MediaType


def create_text_processor(model_manager, model_id: str) -> Processor:
    """Create a text completion processor"""
    from app.models.predictor import Predictor
    backend = DSPyBackend(model_manager, model_id, Predictor)
    return ModelProcessor(
        backend=backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT
    )

def create_business_card_processor(model_manager, model_id: str) -> Processor:
    """Create business card processing pipeline with validation"""
    from app.core.modules import BusinessCardExtractor
    backend = DSPyBackend(model_manager, model_id, BusinessCardExtractor)
    
    pipeline = Pipeline([
        ImageTypeValidator(),
        ImageConverterStep(),
        ModelProcessor(
            backend=backend,
            accepted_types=[MediaType.IMAGE],
            output_type=MediaType.TEXT
        )
    ])
    
    return pipeline

