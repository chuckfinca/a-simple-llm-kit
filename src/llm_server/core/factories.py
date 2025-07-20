from typing import Any, Optional

from llm_server.core.dspy_backend import create_dspy_backend
from llm_server.core.implementations import (
    ImageProcessor,
    ModelProcessor,
)
from llm_server.core.modules import ContactExtractor
from llm_server.core.output_processors import (
    ContactExtractorProcessor,
    DefaultOutputProcessor,
)
from llm_server.core.pipeline import Pipeline
from llm_server.core.protocols import PipelineStep
from llm_server.core.types import MediaType
from llm_server.models.predictor import Predictor


def create_text_processor(
    model_manager,
    model_id: str,
    program_manager=None,
    metadata: Optional[dict[str, Any]] = None,
    output_processor=None,  # New parameter
) -> PipelineStep:
    """Create a text completion processor"""
    backend = create_dspy_backend(
        model_manager,
        model_id,
        Predictor,
        output_processor=output_processor or DefaultOutputProcessor(),
        program_manager=program_manager,
    )
    return ModelProcessor(
        backend=backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT,
        metadata=metadata or {},
    )


def create_extract_contact_processor(
    model_manager,
    model_id: str,
    program_manager=None,
    metadata: Optional[dict[str, Any]] = None,
    output_processor=None,  # New parameter
) -> Pipeline:
    """Create extract contact processing pipeline"""
    # Use the specialized ContactExtractorProcessor by default
    contact_processor = output_processor or ContactExtractorProcessor()

    backend = create_dspy_backend(
        model_manager,
        model_id,
        ContactExtractor,
        output_processor=contact_processor,
        program_manager=program_manager,
    )

    # You can import and use the ImageProcessor from your existing code

    pipeline = Pipeline(
        [
            ImageProcessor(),
            ModelProcessor(
                backend=backend,
                accepted_types=[MediaType.IMAGE],
                output_type=MediaType.TEXT,
                metadata=metadata or {},
            ),
        ]
    )

    return pipeline
