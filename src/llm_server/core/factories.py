from llm_server.core.implementations import (
    DSPyModelBackend,
    ImageProcessor,
    ModelProcessor,
)
from llm_server.core.modules import ContactExtractor
from llm_server.core.output_processors import ContactExtractorProcessor
from llm_server.core.pipeline import Pipeline
from llm_server.core.types import MediaType


def create_extract_contact_processor(model_manager, model_id: str) -> Pipeline:
    """Creates a focused, stateless pipeline for contact extraction."""

    # 1. The backend is created without the program_manager. It's now stateless.
    backend = DSPyModelBackend(
        model_manager=model_manager,
        model_id=model_id,
        signature_class=ContactExtractor
    )

    # 2. The pipeline is assembled with the necessary steps.
    # The robust dual-path logic is in the ContactExtractorProcessor, which is great!
    pipeline = Pipeline(
        [
            ImageProcessor(),
            ModelProcessor(
                backend=backend,
                accepted_types=[MediaType.IMAGE],
                output_type=MediaType.TEXT
            ),
        ]
    )

    return pipeline
