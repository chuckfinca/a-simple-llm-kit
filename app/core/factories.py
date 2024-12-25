from app.core.implementations import DSPyBackend, ModelProcessor
from app.core.modules import OCRModule
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

def create_ocr_processor(model_manager, model_id: str) -> Processor:
    """Create an OCR processor"""
    backend = DSPyBackend(model_manager, model_id, OCRModule)
    return ModelProcessor(
        backend=backend,
        accepted_types=[MediaType.IMAGE],
        output_type=MediaType.TEXT
    )
