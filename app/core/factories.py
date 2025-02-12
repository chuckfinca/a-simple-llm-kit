
from app.core.implementations import DSPyModelBackend, ImageProcessor, ModelProcessor
from app.core.pipeline import Pipeline
from app.core.protocols import PipelineStep
from app.core.types import MediaType
from app.models.predictor import Predictor
from app.core.modules import ContactExtractor

def create_text_processor(model_manager, model_id: str) -> PipelineStep:
    """Create a text completion processor"""
    backend = DSPyModelBackend(model_manager, model_id, Predictor)
    return ModelProcessor(
        backend=backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT
    )

def create_extract_contact_processor(model_manager, model_id: str) -> PipelineStep:
    """Create extract contact processing pipeline"""
    backend = DSPyModelBackend(model_manager, model_id, ContactExtractor)
    
    pipeline = Pipeline([
        ImageProcessor(),
        ModelProcessor(
            backend=backend,
            accepted_types=[MediaType.IMAGE],
            output_type=MediaType.TEXT
        )
    ])
    
    return pipeline