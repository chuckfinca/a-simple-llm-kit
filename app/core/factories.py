from typing import Dict, Any, Optional
from app.core.dspy_backend import create_dspy_backend
from app.core.implementations import DSPyModelBackend, ImageProcessor, ModelProcessor
from app.core.pipeline import Pipeline
from app.core.protocols import PipelineStep
from app.core.types import MediaType
from app.models.predictor import Predictor
from app.core.modules import ContactExtractor

def create_text_processor(
    model_manager, 
    model_id: str, 
    program_manager=None, 
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None  # New parameter
) -> PipelineStep:
    """Create a text completion processor"""
    backend = create_dspy_backend(
        model_manager, 
        model_id, 
        Predictor, 
        output_processor=output_processor,
        program_manager=program_manager
    )
    return ModelProcessor(
        backend=backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT,
        metadata=metadata or {}
    )

def create_extract_contact_processor(
    model_manager, 
    model_id: str, 
    program_manager=None, 
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None  # New parameter
) -> PipelineStep:
    """Create extract contact processing pipeline"""
    backend = create_dspy_backend(
        model_manager, 
        model_id, 
        ContactExtractor, 
        output_processor=output_processor,
        program_manager=program_manager
    )
    
    # You can import and use the ImageProcessor from your existing code
    from app.core.implementations import ImageProcessor
    
    pipeline = Pipeline([
        ImageProcessor(),
        ModelProcessor(
            backend=backend,
            accepted_types=[MediaType.IMAGE],
            output_type=MediaType.TEXT,
            metadata=metadata or {}
        )
    ])
    
    return pipeline
